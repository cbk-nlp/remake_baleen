# 文件名: colbert/data/collection.py

import os
import itertools

# 从评估加载器中导入 load_collection 函数，用于从 tsv 文件加载文档集合
from colbert.evaluation.loaders import load_collection
# 导入 Run 类，用于管理实验的运行环境和路径
from colbert.infra.run import Run


class Collection:
    """
    表示一个文档集合（例如，段落集合）。
    此类可以从文件路径加载数据，也接受一个已经加载的数据列表。
    """

    def __init__(self, path=None, data=None):
        """
        初始化 Collection 对象。

        Args:
            path (str, optional): 文档集合文件的路径 (通常是 .tsv 格式)。
            data (list, optional): 一个包含文档内容的列表。如果提供了 data，则不会从 path 加载。
        """
        self.path = path
        # 如果 data 参数不为空，则直接使用；否则，从 path 指定的文件中加载数据
        self.data = data or self._load_file(path)

    def __iter__(self):
        """使 Collection 对象可迭代。"""
        return self.data.__iter__()

    def __getitem__(self, item):
        """支持通过索引访问集合中的单个文档。"""
        return self.data[item]

    def __len__(self):
        """返回集合中文档的数量。"""
        return len(self.data)

    def _load_file(self, path):
        """
        从文件中加载文档集合。
        目前主要支持 .tsv 文件。
        """
        self.path = path
        return self._load_tsv(path) if path.endswith('.tsv') else self._load_jsonl(path)

    def _load_tsv(self, path):
        """从 .tsv 文件加载文档集合。"""
        return load_collection(path)

    def _load_jsonl(self, path):
        """从 .jsonl 文件加载文档集合 (尚未实现)。"""
        raise NotImplementedError()

    def provenance(self):
        """返回数据来源的路径，用于追踪数据来源。"""
        return self.path
    
    def toDict(self):
        """将对象的来源信息转换为字典格式。"""
        return {'provenance': self.provenance()}

    def save(self, new_path):
        """
        将内存中的文档集合保存到新的 .tsv 文件中。

        Args:
            new_path (str): 保存新文件的路径。
        """
        assert new_path.endswith('.tsv'), "目前仅支持保存为 .tsv 文件。"
        assert not os.path.exists(new_path), f"文件 {new_path} 已存在，无法覆盖。"

        with Run().open(new_path, 'w') as f:
            # 遍历数据，并将每个文档以 "pid\tcontent\n" 的格式写入文件
            for pid, content in enumerate(self.data):
                content = f'{pid}\t{content}\n'
                f.write(content)
            
            return f.name

    def enumerate(self, rank):
        """
        在分布式环境中，按 rank 枚举文档。
        返回一个生成器，产生 (文档全局索引, 文档内容)。
        """
        for _, offset, passages in self.enumerate_batches(rank=rank):
            for idx, passage in enumerate(passages):
                yield (offset + idx, passage)

    def enumerate_batches(self, rank, chunksize=None):
        """
        在分布式环境中，将集合划分为多个批次并按 rank 返回。
        这对于大规模数据的分布式处理非常有用。

        Args:
            rank (int): 当前进程的排名（rank）。
            chunksize (int, optional): 每个数据块的大小。如果为 None，则自动计算。
        
        Returns:
            一个生成器，产生 (块索引, 块内偏移量, 文档块内容)。
        """
        assert rank is not None, "必须提供 rank 参数。"

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        # 使用 itertools.cycle 轮流将数据块分配给不同的进程
        for chunk_idx, owner in enumerate(itertools.cycle(range(Run().nranks))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return
    
    def get_chunksize(self):
        """自动计算合适的块大小，用于分布式处理。"""
        return min(25_000, 1 + len(self) // Run().nranks)

    @classmethod
    def cast(cls, obj):
        """
        一个类方法，用于将不同类型的输入（路径字符串、列表、或 Collection 对象本身）
        统一转换为 Collection 对象，方便 API 调用。
        """
        if type(obj) is str:
            return cls(path=obj)
        if type(obj) is list:
            return cls(data=obj)
        if type(obj) is cls:
            return obj
        assert False, f"无法将类型为 {type(obj)} 的对象转换为 Collection"