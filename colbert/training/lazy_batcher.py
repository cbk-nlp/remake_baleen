# 文件名: colbert/training/lazy_batcher.py

from functools import partial

# 导入 ColBERT 的配置、数据处理和模型分词器模块
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import zipstar
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.data import Collection, Queries, Examples


class LazyBatcher:
    """
    一种“懒惰”的数据批处理器 (Batcher)。

    "懒惰"意味着它只在需要构建一个批次时，才根据三元组中的 ID 去查找
    并加载实际的查询（query）和段落（passage）文本。
    
    这种方式的优点是内存效率高，因为它只需要在内存中保留 ID 列表和
    一个对整个文档集合的引用，而不需要将所有训练样本的文本都加载进来。
    这是 ColBERT 中推荐的、更具扩展性的数据加载方式。
    """

    def __init__(self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1):
        """
        初始化 LazyBatcher。

        Args:
            config (ColBERTConfig): 配置对象。
            triples (str or Examples): 包含三元组ID `(qid, pid+, pid-, ...)` 的文件路径或 Examples 对象。
            queries (str or Queries): 包含查询ID到查询文本映射的文件路径或 Queries 对象。
            collection (str or Collection): 包含段落ID到段落文本映射的文件路径或 Collection 对象。
            rank (int): 当前进程的排名。
            nranks (int): 总进程数。
        """
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway

        # 初始化分词器和张量化函数
        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        # 加载数据 ID，并根据 rank 进行切分
        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        # 加载查询和文档集合的查找字典
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

    def __iter__(self):
        """使对象可迭代。"""
        return self

    def __len__(self):
        """返回当前进程负责处理的三元组数量。"""
        return len(self.triples)

    def __next__(self):
        """
        获取下一个批次的数据。

        它会从 self.triples 中获取一批 ID，然后使用 self.queries 和
        self.collection "懒加载" 相应的文本，最后将它们处理成张量。
        """
        # 计算当前批次的起始和结束位置
        offset, endpos = self.position, self.position + self.bsize
        if endpos > len(self.triples):
            raise StopIteration
        self.position = endpos

        # 获取当前批次的三元组 ID
        batch_triples = self.triples[offset:endpos]

        # 根据 ID 查找文本
        batch_queries = [self.queries[qid] for qid, _, _ in batch_triples]
        batch_passages = [self.collection[pid] for _, p_pos_id, p_neg_id in batch_triples for pid in [p_pos_id, p_neg_id]]
        
        # 假设没有分数
        batch_scores = []
        
        return self.collate(batch_queries, batch_passages, batch_scores)

    def collate(self, queries, passages, scores):
        """将文本列表转换为模型所需的张量批次。"""
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize
        return self.tensorize_triples(queries, passages, scores, self.bsize // self.accumsteps, self.nway)