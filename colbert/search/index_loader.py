# 文件名: colbert/search/index_loader.py

import os
import ujson
import torch

from colbert.utils.utils import print_message
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.search.strided_tensor import StridedTensor


class IndexLoader:
    """
    负责从磁盘加载一个完整的 ColBERT 索引。

    一个 ColBERT 索引主要由以下几部分构成：
    -   Codec: 残差编解码器，包含聚类中心等。
    -   IVF: 倒排文件，用于快速候选生成。
    -   Doclens: 每个文档的长度信息。
    -   Embeddings: 所有被压缩的词元嵌入。
    -   Metadata: 索引的元数据，包含配置等信息。
    """

    def __init__(self, index_path):
        """
        初始化 IndexLoader。

        Args:
            index_path (str): 索引目录的路径。
        """
        self.index_path = index_path

        # 依次加载索引的各个部分
        self._load_codec()
        self._load_ivf()
        self._load_doclens()
        self._load_embeddings()

    def _load_codec(self):
        """加载残差编解码器。"""
        self.codec = ResidualCodec.load(self.index_path)

    def _load_ivf(self):
        """
        加载倒排文件 (IVF)。
        IVF 存储了从聚类中心ID到包含该中心作为最近邻的词元嵌入ID的映射。
        加载后，它被封装在一个 StridedTensor 中以便快速查找。
        """
        ivf, ivf_lengths = torch.load(os.path.join(self.index_path, "ivf.pt"), map_location='cpu')
        self.ivf = StridedTensor(ivf, ivf_lengths)

    def _load_doclens(self):
        """
        从多个 `doclens.*.json` 文件中加载所有文档的长度，并合并。
        """
        doclens = []
        for chunk_idx in range(self.num_chunks):
            doclens_path = os.path.join(self.index_path, f'doclens.{chunk_idx}.json')
            with open(doclens_path) as f:
                chunk_doclens = ujson.load(f)
                doclens.extend(chunk_doclens)
        self.doclens = torch.tensor(doclens)

    def _load_embeddings(self):
        """
        从多个索引块中加载所有被压缩的嵌入向量。
        """
        self.embeddings = ResidualCodec.Embeddings.load_chunks(self.index_path, range(self.num_chunks),
                                                               self.num_embeddings)

    @property
    def metadata(self):
        """延迟加载索引的元数据文件 (metadata.json)。"""
        if not hasattr(self, '_metadata'):
            with open(os.path.join(self.index_path, 'metadata.json')) as f:
                self._metadata = ujson.load(f)
        return self._metadata

    @property
    def num_chunks(self):
        """从元数据中获取索引块的数量。"""
        return self.metadata['num_chunks']

    @property
    def num_embeddings(self):
        """从元数据中获取总的嵌入向量数量。"""
        return self.metadata['num_embeddings']