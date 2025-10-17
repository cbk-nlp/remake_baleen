# 文件名: colbert/searcher.py

import os
import torch
from tqdm import tqdm
from typing import Union

# 导入 ColBERT 的数据处理、模型和索引相关模块
from colbert.data import Collection, Queries, Ranking
from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer
from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig
from colbert.infra.launcher import print_memory_stats

# 为查询定义一个类型别名，可以是多种格式
TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    """
    ColBERT 的主搜索器类。

    该类封装了从加载索引到执行端到端检索的所有功能。
    它负责管理配置、加载模型和索引、编码查询，并最终返回排序的文档列表。
    """

    def __init__(self, index, checkpoint=None, collection=None, config=None):
        """
        初始化 Searcher。

        Args:
            index (str): 要使用的 ColBERT 索引的名称（位于 index_root 下）。
            checkpoint (str, optional): 模型的检查点路径。如果未提供，将尝试从索引的元数据中加载。
            collection (str or Collection, optional): 文档集合的路径或对象。如果未提供，将尝试从配置中加载。
            config (ColBERTConfig, optional): 自定义配置对象。
        """
        print_memory_stats()

        # 合并所有来源的配置
        initial_config = ColBERTConfig.from_existing(config, Run().config)
        default_index_root = initial_config.index_root_
        self.index = os.path.join(default_index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        # 加载模型和索引评分器
        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config).cuda()
        self.ranker = IndexScorer(self.index)

        print_memory_stats()

    def configure(self, **kw_args):
        """用给定的关键字参数更新配置。"""
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries):
        """
        将一个或多个查询文本编码为 ColBERT 嵌入矩阵。

        Args:
            text (TextQueries): 待编码的查询。

        Returns:
            torch.Tensor: 查询的嵌入矩阵。
        """
        queries = text if isinstance(text, list) else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True)
        return Q

    def search(self, text: str, k=10):
        """
        为一个查询字符串执行端到端的搜索。

        Args:
            text (str): 查询字符串。
            k (int, optional): 返回的 top-k 结果数量。默认为 10。

        Returns:
            tuple: (pids, ranks, scores)
        """
        return self.dense_search(self.encode(text), k)

    def search_all(self, queries: TextQueries, k=10):
        """
        为一批查询执行端到端的搜索。

        Args:
            queries (TextQueries): 批量的查询。
            k (int, optional): 每个查询返回的 top-k 结果数量。默认为 10。

        Returns:
            Ranking: 包含所有查询结果的 Ranking 对象。
        """
        queries = Queries.cast(queries)
        queries_ = list(queries.values())
        Q = self.encode(queries_)
        return self._search_all_Q(queries, Q, k)

    def _search_all_Q(self, queries, Q, k):
        """search_all 的辅助函数，处理已编码的查询。"""
        all_scored_pids = [list(zip(*self.dense_search(Q[query_idx:query_idx+1], k=k)))
                           for query_idx in tqdm(range(Q.size(0)), desc="批量搜索")]
        
        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        # 记录来源信息
        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, k=10):
        """
        对于一个已编码的查询，执行检索和重排序。

        Args:
            Q (torch.Tensor): 编码后的查询嵌入矩阵。
            k (int, optional): 返回的 top-k 结果数量。

        Returns:
            tuple:
                - pids (list[int]): 排序后的 top-k 文档 ID。
                - ranks (list[int]): 排名列表 (1 to k)。
                - scores (list[float]): 对应的 ColBERT 分数。
        """
        pids, scores = self.ranker.rank(self.config, Q, k)
        top_k_pids = pids[:k]
        top_k_scores = scores[:k]
        return top_k_pids, list(range(1, len(top_k_pids) + 1)), top_k_scores