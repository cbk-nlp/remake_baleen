# 文件名: colbert/training/rerank_batcher.py

# 导入 ColBERT 的配置、数据处理和 reranker 专用分词器模块
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import flatten, zipstar
from colbert.modeling.reranker.tokenizer import RerankerTokenizer
from colbert.data import Collection, Queries, Examples


class RerankBatcher:
    """
    一个专门为重排序（reranker）模型（如 ElectraReranker）设计的数据批处理器。

    与 `LazyBatcher` 类似，它也采用“懒加载”的方式获取文本。
    不同之处在于它的 `collate` 方法，它会将每个查询与对应的 n-way 段落
    配对，形成 `bsize * nway` 个 (查询, 段落) 对，以适应交叉注意力
    （cross-attention）模型的输入格式。
    """

    def __init__(self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway
        
        assert self.accumsteps == 1, "RerankBatcher 目前不支持梯度累积（accumsteps > 1）"

        # 使用 reranker 专用的分词器
        self.tokenizer = RerankerTokenizer(total_maxlen=config.doc_maxlen, base=config.checkpoint)
        self.position = 0

        # 加载数据
        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        """获取下一个批次的数据。"""
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        # 懒加载文本数据
        for position in range(offset, endpos):
            query_id, *pids_with_scores = self.triples[position]
            pids = pids_with_scores[:self.nway]
            
            query_text = self.queries[query_id]
            
            try: # 尝试解析 (pid, score) 对
                pids, scores = zipstar(pids)
            except: # 如果只有 pid
                scores = []

            passages = [self.collection[pid] for pid in pids]

            all_queries.append(query_text)
            all_passages.extend(passages)
            all_scores.extend(scores)
        
        return self.collate(all_queries, all_passages, all_scores)

    def collate(self, queries, passages, scores):
        """
        将数据整理为 reranker 模型所需的输入格式。
        """
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize

        # 将每个查询复制 nway 次，与对应的 nway 个段落配对
        queries_paired = flatten([[query] * self.nway for query in queries])
        
        # 返回一个只包含一个步骤的批次列表
        return [(self.tokenizer.tensorize(queries_paired, passages), scores)]