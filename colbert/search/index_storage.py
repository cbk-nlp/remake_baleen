# 文件名: colbert/search/index_storage.py

import torch

from colbert.utils.utils import flatten
from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from colbert.search.strided_tensor import StridedTensor
from colbert.search.candidate_generation import CandidateGeneration
from .index_loader import IndexLoader
from colbert.modeling.colbert import colbert_score, colbert_score_packed


class IndexScorer(IndexLoader, CandidateGeneration):
    """
    一个集成了索引加载、候选生成和精确评分功能的核心类。

    它继承了 `IndexLoader` 来加载索引数据，继承了 `CandidateGeneration` 来执行
    第一阶段的检索。此外，它还实现了第二阶段的精确评分（重排序）逻辑。
    """

    def __init__(self, index_path):
        """
        初始化 IndexScorer。
        """
        super().__init__(index_path)

        # 创建一个支持按 PID 快速查找的 Strided 嵌入对象
        self.embeddings_strided = ResidualEmbeddingsStrided(self.codec, self.embeddings, self.doclens)

        # 构建一个从全局嵌入 ID (eid) 到文档 ID (pid) 的映射
        # 这个映射对于从候选 eids 中找到对应的 pids 至关重要
        all_doclens_nested = load_doclens(index_path, flatten=False)
        all_doclens_flat = flatten(all_doclens_nested)
        self.emb2pid = torch.zeros(self.num_embeddings, dtype=torch.int)
        
        offset = 0
        for pid, dlength in enumerate(all_doclens_flat):
            self.emb2pid[offset: offset + dlength] = pid
            offset += dlength

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        """根据全局嵌入 ID (eids) 查找并解压缩嵌入。"""
        return self.embeddings_strided.lookup_eids(embedding_ids, codes=codes, out_device=out_device)

    def lookup_pids(self, passage_ids, out_device='cuda', return_mask=False):
        """根据文档 ID (pids) 查找并解压缩这些文档的所有嵌入。"""
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

    def retrieve(self, config, Q):
        """
        执行检索的第一阶段：候选生成。
        返回一个候选文档 ID (PID) 的列表。
        """
        # 注意：候选生成只使用查询的实际内容，不使用 [MASK] 填充部分
        Q_subset = Q[:, :config.query_maxlen]
        pids = self.generate_candidates(config, Q_subset)
        return pids

    def rank(self, config, Q, k):
        """
        执行完整的两阶段检索和排序流程。

        Args:
            config (ColBERTConfig): 配置对象。
            Q (torch.Tensor): 编码后的查询。
            k (int): 需要返回的 top-k 结果数。

        Returns:
            tuple: (pids, scores)
        """
        with torch.inference_mode():
            # 阶段一：检索候选 PID
            pids = self.retrieve(config, Q)
            # 阶段二：对候选 PID 进行精确评分
            scores = self.score_pids(config, Q, pids, k)

            # 对分数进行排序并返回 top-k
            scores_sorter = scores.sort(descending=True)
            pids, scores = pids[scores_sorter.indices], scores_sorter.values
            return pids.tolist(), scores.tolist()

    def score_pids(self, config, Q, pids, k):
        """
        对一批候选文档 (由 pids 指定) 进行精确的 ColBERT (MaxSim) 评分。
        """
        if len(pids) == 0:
            return torch.tensor([])
            
        # 1. 从索引中查找并解压缩这些 PID 对应的所有嵌入
        D_packed, D_mask = self.lookup_pids(pids)

        # 2. 调用核心的 colbert_score 函数计算分数
        if Q.size(0) == 1:
            return colbert_score_packed(Q, D_packed, D_mask, config)
        
        # 处理批量查询的情况 (虽然在当前 Searcher 中不常用)
        D_padded, _ = StridedTensor(D_packed, D_mask).as_padded_tensor()
        return colbert_score(Q, D_padded, D_mask.sum(-1), config)