# 文件名: colbert/indexing/codecs/residual_embeddings.py

import os
import ujson
import torch

from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided


class ResidualEmbeddings:
    """
    一个数据结构，用于封装压缩后的嵌入向量。
    它包含两部分：
    1.  codes: 每个原始嵌入向量对应的最近聚类中心的索引。
    2.  residuals: 经过二值化编码后的残差向量。
    """
    Strided = ResidualEmbeddingsStrided

    def __init__(self, codes, residuals):
        """
        初始化 ResidualEmbeddings。

        Args:
            codes (torch.Tensor): 形状为 (num_embeddings,) 的整数张量。
            residuals (torch.Tensor): 形状为 (num_embeddings, compressed_dim) 的 uint8 张量。
        """
        assert codes.size(0) == residuals.size(0)
        assert codes.dim() == 1 and residuals.dim() == 2
        assert residuals.dtype == torch.uint8

        self.codes = codes.to(torch.int32)
        self.residuals = residuals

    @classmethod
    def load_chunks(cls, index_path, chunk_idxs, num_embeddings):
        """
        从磁盘加载多个索引块（chunks）并将它们合并到一个 ResidualEmbeddings 对象中。
        这在检索时用于将整个索引加载到内存中。
        """
        # 为可能存在的步长访问添加一些填充
        num_embeddings += 512
        dim, nbits = get_dim_and_nbits(index_path)

        # 预分配内存
        codes = torch.empty(num_embeddings, dtype=torch.int32)
        residuals = torch.empty(num_embeddings, dim // 8 * nbits, dtype=torch.uint8)

        codes_offset = 0
        for chunk_idx in chunk_idxs:
            chunk = cls.load(index_path, chunk_idx)
            codes_endpos = codes_offset + chunk.codes.size(0)

            # 将加载的块数据复制到预分配的张量中
            codes[codes_offset:codes_endpos] = chunk.codes
            residuals[codes_offset:codes_endpos] = chunk.residuals
            codes_offset = codes_endpos

        return cls(codes, residuals)

    @classmethod
    def load(cls, index_path, chunk_idx):
        """加载单个索引块。"""
        codes = cls.load_codes(index_path, chunk_idx)
        residuals = cls.load_residuals(index_path, chunk_idx)
        return cls(codes, residuals)

    @classmethod
    def load_codes(cls, index_path, chunk_idx):
        """加载单个索引块的 codes 部分。"""
        codes_path = os.path.join(index_path, f'{chunk_idx}.codes.pt')
        return torch.load(codes_path, map_location='cpu')

    @classmethod
    def load_residuals(self, index_path, chunk_idx):
        """加载单个索引块的 residuals 部分。"""
        residuals_path = os.path.join(index_path, f'{chunk_idx}.residuals.pt')
        return torch.load(residuals_path, map_location='cpu')

    def save(self, path_prefix):
        """将 codes 和 residuals 保存到磁盘。"""
        codes_path = f'{path_prefix}.codes.pt'
        residuals_path = f'{path_prefix}.residuals.pt'
        torch.save(self.codes, codes_path)
        torch.save(self.residuals, residuals_path)

    def __len__(self):
        """返回嵌入向量的数量。"""
        return self.codes.size(0)


def get_dim_and_nbits(index_path):
    """
    从索引元数据文件中读取嵌入维度 (dim) 和压缩位数 (nbits)。
    """
    with open(os.path.join(index_path, 'metadata.json')) as f:
        metadata = ujson.load(f)['config']
    dim = metadata['dim']
    nbits = metadata['nbits']
    assert (dim * nbits) % 8 == 0, "维度和比特数的乘积必须是8的倍数"
    return dim, nbits