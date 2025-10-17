# 文件名: colbert/training/eager_batcher.py

from functools import partial

# 导入 ColBERT 的模型分词器和张量化工具
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
# 导入 Run 类，用于日志记录
from colbert.infra.run import Run


class EagerBatcher:
    """
    一种“急切”的数据批处理器 (Batcher)。

    "急切"意味着它会尝试在迭代开始时一次性从文件中读取整个批次的数据。
    这种方式适用于训练数据不大，可以直接从一个简单的制表符分隔（TSV）文件中流式读取的场景。
    每一行代表一个三元组 (query, positive_passage, negative_passage)。

    与 `LazyBatcher` 相比，`EagerBatcher` 不需要在内存中维护整个文档集合（collection）
    的索引，因为它假设段落文本直接包含在三元组文件中。
    """

    def __init__(self, args, rank=0, nranks=1):
        """
        初始化 EagerBatcher。

        Args:
            args (ColBERTConfig): 包含所有配置参数的对象。
            rank (int, optional): 当前进程的排名，用于分布式数据并行。
            nranks (int, optional): 总进程数。
        """
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        # 初始化查询和文档的分词器
        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        # 使用 functools.partial 创建一个预设了分词器参数的 tensorize_triples 函数
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)

        self.triples_path = args.triples
        self._reset_triples()

    def _reset_triples(self):
        """重置文件读取器，使其回到文件开头。"""
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0

    def __iter__(self):
        """使对象可迭代。"""
        return self

    def __next__(self):
        """
        获取下一个批次的数据。

        它会从文件中读取 `bsize` 行，并将它们处理成模型所需的张量格式。
        在分布式环境中，每个 rank 只会处理属于自己的那部分数据行。
        """
        queries, positives, negatives = [], [], []

        # 尝试读取 self.bsize * self.nranks 行，以确保每个进程都能获得 self.bsize 个样本
        # （这是一个近似的实现，不完全精确，但简单有效）
        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            # 根据 rank 决定是否处理当前行
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            query, pos, neg = line.strip().split('\t')
            queries.append(query)
            positives.append(pos)
            negatives.append(neg)

        # 更新文件读取的全局位置
        self.position += line_idx + 1

        # 如果读取到的样本数少于一个批次的大小，说明数据已读完
        if len(queries) < self.bsize:
            raise StopIteration

        # 将文本数据整理并转换为张量
        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        """将文本列表转换为模型所需的张量批次。"""
        assert len(queries) == len(positives) == len(negatives) == self.bsize
        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        """
        跳过文件的前 N 个批次。用于从中断的训练中恢复。
        """
        self._reset_triples()
        Run.warn(f'正在跳至训练的批次 #{batch_idx}...')
        
        # 逐行读取并丢弃，直到达到目标位置
        for _ in range(batch_idx * intended_batch_size):
            self.reader.readline()