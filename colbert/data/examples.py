# 文件名: colbert/data/examples.py

import os
import ujson

# 导入 Run 类，用于管理实验的运行环境和文件操作
from colbert.infra.run import Run
# 导入 print_message 工具函数，用于格式化输出信息
from colbert.utils.utils import print_message
# 导入 Provenance 类，用于追踪数据的来源和生成过程
from colbert.infra.provenance import Provenance
# 导入 get_metadata_only 工具函数，用于获取运行环境的元数据
from utility.utils.save_metadata import get_metadata_only


class Examples:
    """
    表示一个训练样本集合，通常是三元组 (query, positive_passage, negative_passage)。
    此类可以从文件加载数据，也可以直接使用内存中的数据列表。
    """

    def __init__(self, path=None, data=None, nway=None, provenance=None):
        """
        初始化 Examples 对象。

        Args:
            path (str, optional): 训练样本文件的路径 (通常是 .json 格式)。
            data (list, optional): 一个包含训练样本的列表。
            nway (int, optional): 指定每个样本中包含的段落数量 (例如，对于三元组是2-way)。
            provenance (Provenance, optional): 数据的来源信息对象。
        """
        self.__provenance = provenance or path or Provenance()
        self.nway = nway
        self.path = path
        self.data = data or self._load_file(path)

    def provenance(self):
        """返回数据来源信息。"""
        return self.__provenance
    
    def toDict(self):
        """将对象的来源信息转换为字典格式。"""
        return self.provenance()

    def _load_file(self, path):
        """
        从 .json 文件中加载训练样本。
        每行是一个 JSON 对象，代表一个样本。
        """
        nway = self.nway + 1 if self.nway else self.nway
        examples = []

        with open(path) as f:
            for line in f:
                # 使用 ujson 解析每行的 JSON 数据，并根据 nway 进行切片
                example = ujson.loads(line)[:nway]
                examples.append(example)

        return examples

    def tolist(self, rank=None, nranks=None):
        """
        将样本数据转换为列表。在分布式训练中，可以根据 rank 和 nranks 对数据进行切分。

        注意: 这种切分方式不是严格的均匀采样，但由于数据文件本身是预先打乱的，
        并且在训练中不会重复遍历数据，因此可以保证每个进程处理的是不同的随机子集。

        Args:
            rank (int, optional): 当前进程的排名。
            nranks (int, optional): 总进程数。

        Returns:
            list: 样本列表或其子集。
        """
        if rank is not None and nranks is not None:
            assert rank in range(nranks), (rank, nranks)
            # 简单的步长切分，为每个进程分配不同的数据
            return [self.data[idx] for idx in range(rank, len(self.data), nranks)]

        return list(self.data)

    def save(self, new_path):
        """
        将内存中的训练样本保存到新的 .json 文件中。

        Args:
            new_path (str): 保存新文件的路径。
        """
        assert 'json' in new_path.strip('/').split('/')[-1].split('.'), "目前仅支持保存为 .json 或 .jsonl 文件。"

        print_message(f"#> 正在将 {len(self.data) / 1000_000.0}M 个样本写入 {new_path}")

        with Run().open(new_path, 'w') as f:
            for example in self.data:
                ujson.dump(example, f)
                f.write('\n')

            output_path = f.name
            print_message(f"#> 已将 {len(self.data)} 行样本保存至 {f.name}")
        
        # 同时保存一个 .meta 文件，记录元数据和来源信息
        with Run().open(f'{new_path}.meta', 'w') as f:
            d = {}
            d['metadata'] = get_metadata_only()
            d['provenance'] = self.provenance()
            line = ujson.dumps(d, indent=4)
            f.write(line)

        return output_path

    @classmethod
    def cast(cls, obj, nway=None):
        """
        一个类方法，用于将不同类型的输入（路径字符串、列表、或 Examples 对象本身）
        统一转换为 Examples 对象。
        """
        if type(obj) is str:
            return cls(path=obj, nway=nway)
        if isinstance(obj, list):
            return cls(data=obj, nway=nway)
        if type(obj) is cls:
            assert nway is None, "当输入已经是 Examples 对象时，不应再提供 nway 参数"
            return obj
        assert False, f"无法将类型为 {type(obj)} 的对象转换为 Examples"