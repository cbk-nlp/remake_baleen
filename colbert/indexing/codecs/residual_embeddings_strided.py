# 文件名: colbert/indexing/codecs/residual_embeddings_strided.py

# 导入 colbert.indexing.codecs.residual_embeddings 模块，防止循环导入
import colbert.indexing.codecs.residual_embeddings as residual_embeddings
# 导入 StridedTensor，这是一个用于高效处理变长序列的核心数据结构
from colbert.search.strided_tensor import StridedTensor


class ResidualEmbeddingsStrided:
    """
    扩展了 ResidualEmbeddings，增加了对步长张量 (StridedTensor) 的支持。

    StridedTensor 是一种内存布局优化的数据结构，它允许我们快速地通过文档ID (pid)
    来索引和提取该文档包含的所有（变长的）词元嵌入向量。
    这对于检索时的文档评分至关重要。
    """

    def __init__(self, codec, embeddings, doclens):
        """
        初始化 ResidualEmbeddingsStrided。

        Args:
            codec (ResidualCodec): 残差编解码器实例。
            embeddings (ResidualEmbeddings): 包含所有压缩嵌入的对象。
            doclens (torch.Tensor): 一个张量，包含每个文档的长度（即词元数量）。
        """
        self.codec = codec
        self.codes = embeddings.codes
        self.residuals = embeddings.residuals

        # 将扁平的 codes 和 residuals 数组与 doclens 结合，创建 StridedTensor
        # 这样就可以通过文档 ID 来高效地进行索引
        self.codes_strided = StridedTensor(self.codes, doclens)
        self.residuals_strided = StridedTensor(self.residuals, doclens)

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        """
        根据全局的嵌入ID (eid) 查找并解压缩嵌入向量。
        主要用于 FAISS 检索的第一阶段（召回候选）。
        """
        # 如果未提供 codes，则从 self.codes 中查找
        codes = self.codes[embedding_ids] if codes is None else codes
        residuals = self.residuals[embedding_ids]

        # 使用编解码器进行解压缩
        return self.codec.decompress(residual_embeddings.ResidualEmbeddings(codes, residuals))

    def lookup_pids(self, passage_ids, out_device='cuda'):
        """
        根据文档ID (pid) 列表查找并解压缩这些文档的所有嵌入向量。
        这是检索时对候选文档进行重排序（re-ranking）的关键步骤。

        Args:
            passage_ids (list or torch.Tensor): 需要查找的文档ID列表。

        Returns:
            tuple:
                - embeddings_packed (torch.Tensor): 解压缩后的嵌入向量，以扁平化（packed）形式存储。
                - codes_lengths (torch.Tensor): 每个文档对应的嵌入向量数量。
        """
        # 使用 StridedTensor 高效地根据 pid 提取出对应的 codes 和 residuals
        codes_packed, codes_lengths = self.codes_strided.lookup(passage_ids)
        residuals_packed, _ = self.residuals_strided.lookup(passage_ids)

        # 解压缩提取出的 codes 和 residuals
        embeddings_packed = self.codec.decompress(residual_embeddings.ResidualEmbeddings(codes_packed, residuals_packed))

        return embeddings_packed, codes_lengths