# 文件名: colbert/data/ranking.py

import os
import tqdm
import ujson

# 导入 Provenance 类，用于追踪数据来源
from colbert.infra.provenance import Provenance
# 导入 Run 类，用于管理实验环境和文件操作
from colbert.infra.run import Run
# 导入工具函数，用于格式化打印信息和按 qid 分组
from colbert.utils.utils import print_message, groupby_first_item
# 导入工具函数，用于获取元数据
from utility.utils.save_metadata import get_metadata_only


def numericize(v):
    """将字符串转换为整数或浮点数。"""
    if '.' in v:
        return float(v)
    return int(v)


def load_ranking(path):
    """
    从 .tsv 文件中加载排序列表。
    文件每行格式通常为: qid, pid, rank, score, [label]
    """
    print_message("#> 正在从", path, "加载排序列表...")
    with open(path) as f:
        return [list(map(numericize, line.strip().split('\t'))) for line in f]


class Ranking:
    """
    表示一个排序结果集合。
    可以从文件加载，也可以直接使用内存中的数据。
    """

    def __init__(self, path=None, data=None, metrics=None, provenance=None):
        """
        初始化 Ranking 对象。

        Args:
            path (str, optional): 排序结果文件的路径。
            data (list or dict, optional): 包含排序结果的数据。
            metrics (any, optional): 与排序相关的评估指标 (目前未使用)。
            provenance (Provenance, optional): 数据的来源信息对象。
        """
        self.__provenance = provenance or path or Provenance()
        self.data = self._prepare_data(data or self._load_file(path))

    def provenance(self):
        """返回数据来源信息。"""
        return self.__provenance
    
    def toDict(self):
        """将对象的来源信息转换为字典格式。"""
        return {'provenance': self.provenance()}

    def _prepare_data(self, data):
        """
        准备数据。如果数据是列表，则按 qid 分组为字典；
        如果已经是字典，则直接使用并生成扁平化列表。
        """
        if isinstance(data, dict):
            # 从字典生成扁平化列表
            self.flat_ranking = [(qid, *rest) for qid, subranking in data.items() for rest in subranking]
            return data

        # 从列表生成字典
        self.flat_ranking = data
        return groupby_first_item(tqdm.tqdm(self.flat_ranking, desc="按 QID 分组排序结果"))

    def _load_file(self, path):
        """从文件加载排序数据。"""
        return load_ranking(path)

    def todict(self):
        """返回按 qid 分组的字典。"""
        return dict(self.data)

    def tolist(self):
        """返回扁平化的排序结果列表。"""
        return list(self.flat_ranking)

    def items(self):
        """返回字典的 (key, value) 对。"""
        return self.data.items()

    def save(self, new_path):
        """
        将排序结果保存到新的 .tsv 文件中。

        Args:
            new_path (str): 保存文件的路径。
        """
        assert 'tsv' in new_path.strip('/').split('/')[-1].split('.'), "目前仅支持保存为 .tsv 文件。"

        with Run().open(new_path, 'w') as f:
            for items in self.flat_ranking:
                # 将布尔值转换为整数，然后所有项转换为字符串
                line = '\t'.join(map(lambda x: str(int(x) if type(x) is bool else x), items)) + '\n'
                f.write(line)

            output_path = f.name
            print_message(f"#> 已将 {len(self.data)} 个查询的 {len(self.flat_ranking)} 行排序结果保存至 {f.name}")
        
        # 同时保存一个 .meta 文件，记录元数据和来源信息
        with Run().open(f'{new_path}.meta', 'w') as f:
            d = {}
            d['metadata'] = get_metadata_only()
            d['provenance'] = self.provenance()
            line = ujson.dumps(d, indent=4)
            f.write(line)
        
        return output_path

    @classmethod
    def cast(cls, obj):
        """
        一个类方法，用于将不同类型的输入（路径字符串、字典、列表或 Ranking 对象本身）
        统一转换为 Ranking 对象。
        """
        if type(obj) is str:
            return cls(path=obj)
        if isinstance(obj, dict) or isinstance(obj, list):
            return cls(data=obj)
        if type(obj) is cls:
            return obj
        assert False, f"无法将类型为 {type(obj)} 的对象转换为 Ranking"