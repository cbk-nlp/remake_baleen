# 文件名: baleen/hop_searcher.py

from typing import Union
from colbert import Searcher
from colbert.data import Queries
from colbert.infra.config import ColBERTConfig

# 定义类型别名
TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class HopSearcher(Searcher):
    """
    一个为多跳（multi-hop）搜索定制的搜索器。

    它继承自 `colbert.Searcher`，并重写了 `encode` 和 `search` 方法，
    使其能够接受一个额外的 `context` 参数。
    在多跳问答中，这个 `context` 通常是前几跳中找到的关键信息片段（facts），
    它被拼接到当前查询上，以形成一个更具信息量的新查询。
    """
    def __init__(self, *args, config=None, **kw_args):
        # 设置多跳搜索的默认配置，例如更长的查询最大长度
        defaults = ColBERTConfig(query_maxlen=64)
        config = ColBERTConfig.from_existing(defaults, config)
        super().__init__(*args, config=config, **kw_args)

    def encode(self, text: TextQueries, context: TextQueries = None):
        """
        编码查询，同时可以选择性地拼接上下文。

        Args:
            text (TextQueries): 主查询文本。
            context (TextQueries, optional): 上下文文本。

        Returns:
            torch.Tensor: 编码后的查询嵌入矩阵。
        """
        queries = text if isinstance(text, list) else [text]
        contexts = context if context is None or isinstance(context, list) else [context]
        
        # 调用底层的 Checkpoint.queryFromText 方法，它支持 context 参数
        Q = self.checkpoint.queryFromText(queries, context=contexts, bsize=128, to_cpu=True)
        return Q

    def search(self, text: str, context: str = None, k=10):
        """
        执行一次带上下文的搜索。

        Args:
            text (str): 主查询。
            context (str, optional): 上下文。
            k (int, optional): 返回的 top-k 结果数量。

        Returns:
            tuple: (pids, ranks, scores)
        """
        # 1. 编码带上下文的查询
        Q = self.encode(text, context)
        # 2. 执行标准的密集检索
        return self.dense_search(Q, k)

    def search_all(self, queries: TextQueries, context: TextQueries, k=10):
        """对一批带上下文的查询进行搜索。"""
        # ... (与基类类似，但处理 context) ...
        queries_obj = Queries.cast(queries)
        context_obj = Queries.cast(context) if context is not None else None
        
        Q = self.encode(list(queries_obj.values()), list(context_obj.values()) if context_obj else None)
        return self._search_all_Q(queries_obj, Q, k)