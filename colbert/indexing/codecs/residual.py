# 文件名: colbert/indexing/codecs/residual.py

import os
import cupy  # 用于高效的位操作
import torch

from colbert.infra.config import ColBERTConfig
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings


class ResidualCodec:
    """
    残差编解码器 (Residual Codec)。

    该类实现了 ColBERT 的核心压缩算法。它通过以下步骤工作：
    1.  **聚类中心**: 空间被划分为多个区域，每个区域由一个“聚类中心”（centroid）向量代表。
    2.  **量化**: 对于每个文档的词元嵌入（embedding），找到距离最近的聚类中心。
    3.  **残差计算**: 计算原始嵌入与最近聚类中心之间的差值，即“残差”（residual）。
    4.  **残差编码**: 对残差进行进一步的量化和二值化编码，以极大地压缩其存储大小。

    在检索时，通过解码残差并将其加回到对应的聚类中心，可以近似地重构出原始的嵌入向量。
    """
    Embeddings = ResidualEmbeddings

    def __init__(self, config, centroids, avg_residual=None, bucket_cutoffs=None, bucket_weights=None):
        """
        初始化 ResidualCodec。

        Args:
            config (ColBERTConfig): 配置对象。
            centroids (torch.Tensor): K-means 聚类得到的中心点张量。
            avg_residual (torch.Tensor, optional): 平均残差值，用于训练阶段。
            bucket_cutoffs (torch.Tensor, optional): 用于残差量化的桶边界。
            bucket_weights (torch.Tensor, optional): 每个桶的重建权重。
        """
        self.dim, self.nbits = config.dim, config.nbits
        self.centroids = centroids.half().cuda()  # 使用半精度以节省 GPU 显存
        self.avg_residual = avg_residual

        if torch.is_tensor(self.avg_residual):
            self.avg_residual = self.avg_residual.half().cuda()
        
        if torch.is_tensor(bucket_cutoffs):
            bucket_cutoffs = bucket_cutoffs.cuda()
            bucket_weights = bucket_weights.half().cuda()

        self.bucket_cutoffs = bucket_cutoffs
        self.bucket_weights = bucket_weights

        # 用于二值化操作的辅助张量
        self.arange_bits = torch.arange(0, self.nbits, device='cuda', dtype=torch.uint8)

    @classmethod
    def load(cls, index_path):
        """从索引目录加载编解码器所需的所有组件。"""
        config = ColBERTConfig.load_from_index(index_path)
        centroids_path = os.path.join(index_path, 'centroids.pt')
        avgresidual_path = os.path.join(index_path, 'avg_residual.pt')
        buckets_path = os.path.join(index_path, 'buckets.pt')

        centroids = torch.load(centroids_path, map_location='cpu')
        avg_residual = torch.load(avgresidual_path, map_location='cpu')
        bucket_cutoffs, bucket_weights = torch.load(buckets_path, map_location='cpu')

        if avg_residual.dim() == 0:
            avg_residual = avg_residual.item()

        return cls(config=config, centroids=centroids, avg_residual=avg_residual, bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights)

    def save(self, index_path):
        """将编解码器的所有组件保存到索引目录。"""
        assert self.avg_residual is not None
        assert torch.is_tensor(self.bucket_cutoffs)
        assert torch.is_tensor(self.bucket_weights)

        centroids_path = os.path.join(index_path, 'centroids.pt')
        avgresidual_path = os.path.join(index_path, 'avg_residual.pt')
        buckets_path = os.path.join(index_path, 'buckets.pt')

        torch.save(self.centroids, centroids_path)
        torch.save((self.bucket_cutoffs, self.bucket_weights), buckets_path)
        torch.save(torch.tensor([self.avg_residual]) if not torch.is_tensor(self.avg_residual) else self.avg_residual, avgresidual_path)

    def compress(self, embs):
        """
        压缩一批嵌入向量。

        Args:
            embs (torch.Tensor): 形状为 (num_embeddings, dim) 的浮点型嵌入张量。

        Returns:
            ResidualEmbeddings: 包含压缩后的 codes 和 residuals 的对象。
        """
        codes, residuals = [], []

        # 分批处理以避免 GPU OOM
        for batch in embs.split(1 << 18):
            batch = batch.cuda().half()
            codes_ = self.compress_into_codes(batch, out_device=batch.device)
            centroids_ = self.lookup_centroids(codes_, out_device=batch.device)

            residuals_ = (batch - centroids_)

            codes.append(codes_.cpu())
            residuals.append(self.binarize(residuals_).cpu())

        return self.Embeddings(torch.cat(codes), torch.cat(residuals))

    def binarize(self, residuals):
        """
        对残差进行二值化编码。
        这是压缩的关键步骤，将浮点数的残差向量转换为紧凑的二进制表示。
        """
        # 1. 将浮点残差量化到预定义的桶中
        residuals = torch.bucketize(residuals.float(), self.bucket_cutoffs).to(dtype=torch.uint8)
        # 2. 扩展维度以准备位操作
        residuals = residuals.unsqueeze(-1).expand(*residuals.size(), self.nbits)
        # 3. 通过位移和与操作提取每个比特位
        residuals = (residuals >> self.arange_bits) & 1
        
        # 4. 使用 CuPy 的 packbits 将二进制位打包成字节，极大提高效率
        residuals_packed = cupy.packbits(cupy.asarray(residuals.contiguous().flatten()))
        residuals_packed = torch.as_tensor(residuals_packed, dtype=torch.uint8)
        
        # 5. 重塑为最终的紧凑形状
        return residuals_packed.reshape(residuals.size(0), self.dim // 8 * self.nbits)

    def compress_into_codes(self, embs, out_device):
        """
        对于每个嵌入向量，找到其最近的聚类中心的索引 (code)。
        这通过计算嵌入与所有中心点的矩阵乘法并取最大值实现。
        """
        codes = []
        bsize = (1 << 29) // self.centroids.size(0)  # 动态计算批次大小以控制内存
        for batch in embs.split(bsize):
            # (centroids @ batch.T) 计算所有嵌入与所有中心点的相似度
            indices = (self.centroids @ batch.T.cuda().half()).max(dim=0).indices.to(device=out_device)
            codes.append(indices)
        return torch.cat(codes)

    def lookup_centroids(self, codes, out_device):
        """根据 codes (索引) 查找对应的聚类中心向量。"""
        centroids = []
        for batch in codes.split(1 << 20):
            centroids.append(self.centroids[batch.cuda().long()].to(device=out_device))
        return torch.cat(centroids)

    def decompress(self, compressed_embs: Embeddings):
        """
        解压缩嵌入向量。

        Args:
            compressed_embs (ResidualEmbeddings): 包含 codes 和 residuals 的压缩对象。

        Returns:
            torch.Tensor: 解压缩并归一化后的嵌入向量。
        """
        codes, residuals = compressed_embs.codes, compressed_embs.residuals
        D = []
        # 分批处理以避免 OOM
        for codes_, residuals_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):
            codes_, residuals_ = codes_.cuda(), residuals_.cuda()
            centroids_ = self.lookup_centroids(codes_, out_device='cuda')
            residuals_ = self.decompress_residuals(residuals_).to(device=centroids_.device)
            
            # 重构嵌入 = 聚类中心 + 残差
            reconstructed_embs = centroids_.add_(residuals_)
            
            # 最终进行 L2 归一化
            D.append(torch.nn.functional.normalize(reconstructed_embs, p=2, dim=-1).half())
        
        return torch.cat(D)

    def decompress_residuals(self, binary_residuals):
        """
        从二值化表示中解压缩残差。
        这是 `binarize` 的逆过程。
        """
        # 1. 使用 CuPy 的 unpackbits 将字节解包成二进制位
        residuals = cupy.unpackbits(cupy.asarray(binary_residuals.contiguous().flatten()))
        residuals = torch.as_tensor(residuals, dtype=torch.uint8, device='cuda')
        
        # 2. 如果 nbits > 1, 重建多比特的量化索引
        if self.nbits > 1:
            residuals = residuals.reshape(binary_residuals.size(0), self.dim, self.nbits)
            residuals = (residuals << self.arange_bits).sum(-1)
        
        # 3. 使用桶权重 (bucket_weights) 将量化索引映射回浮点数值
        residuals = residuals.reshape(binary_residuals.size(0), self.dim)
        return self.bucket_weights[residuals.long()].cuda()