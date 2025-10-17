# 文件名: colbert/infra/config/core_config.py

import os
import ujson
import dataclasses
from dataclasses import dataclass, fields
from typing import Any

# 导入工具函数，用于从检查点加载模型配置
from colbert.utils.utils import torch_load_dnn
# 导入工具函数，用于获取元数据
from utility.utils.save_metadata import get_metadata_only


@dataclass
class DefaultVal:
    """一个简单的包装类，用于区分用户未设置的默认值和用户显式设置的 None 值。"""
    val: Any


@dataclass
class CoreConfig:
    """
    配置类的核心基类。

    它使用 Python 的 dataclasses 特性来定义配置项，并提供了一系列
    方法来管理这些配置，例如：
    - 自动处理默认值。
    - 从关键字参数动态配置。
    - 打印帮助信息。
    - 导出配置为字典。
    """

    def __post_init__(self):
        """
        在对象初始化后自动调用的方法。
        
        它会遍历所有的字段，将使用 DefaultVal 包装的默认值或 None 值
        替换为真正的默认值，并记录哪些字段已被赋值。
        """
        self.assigned = {}
        for field in fields(self):
            field_val = getattr(self, field.name)
            if isinstance(field_val, DefaultVal) or field_val is None:
                # 如果字段值是 DefaultVal 或 None，则设置为字段定义的默认值
                setattr(self, field.name, field.default.val)
            if not isinstance(field_val, DefaultVal):
                # 记录该字段已被赋值
                self.assigned[field.name] = True
    
    def assign_defaults(self):
        """强制为所有字段分配它们的默认值。"""
        for field in fields(self):
            setattr(self, field.name, field.default.val)
            self.assigned[field.name] = True

    def configure(self, ignore_unrecognized=True, **kw_args):
        """
        使用关键字参数来配置对象的属性。

        Args:
            ignore_unrecognized (bool, optional): 如果为 True，则忽略无法识别的参数。
                                                  如果为 False，则在遇到无法识别的参数时抛出异常。
                                                  默认为 True。
            **kw_args: 任意数量的关键字参数，用于设置配置项。

        Returns:
            set: 一个包含所有被忽略的参数名的集合。
        """
        ignored = set()
        for key, value in kw_args.items():
            if not self.set(key, value, ignore_unrecognized):
                ignored.add(key)
        return ignored

    def set(self, key, value, ignore_unrecognized=False):
        """
        设置单个配置项的值。

        Args:
            key (str): 配置项的名称。
            value (any): 要设置的值。
            ignore_unrecognized (bool, optional): 是否忽略无法识别的配置项。默认为 False。
        """
        if hasattr(self, key):
            setattr(self, key, value)
            self.assigned[key] = True
            return True
        if not ignore_unrecognized:
            raise Exception(f"无法识别的配置项 `{key}` (对于类型 {type(self)})")
        return False

    def help(self):
        """以格式化的 JSON 形式打印当前的所有配置项及其值。"""
        print(ujson.dumps(dataclasses.asdict(self), indent=4))

    def __export_value(self, v):
        """辅助函数，用于在导出配置时处理大型列表或字典，避免输出过长。"""
        if hasattr(v, 'provenance'):
            v = v.provenance()
        if isinstance(v, list) and len(v) > 100:
            v = (f"一个包含 {len(v)} 个元素的列表，前三个为...", v[:3])
        if isinstance(v, dict) and len(v) > 100:
            v = (f"一个包含 {len(v)} 个键的字典，前三个为...", list(v.keys())[:3])
        return v

    def export(self):
        """将当前配置导出为一个字典。"""
        d = dataclasses.asdict(self)
        for k, v in d.items():
            d[k] = self.__export_value(v)
        return d