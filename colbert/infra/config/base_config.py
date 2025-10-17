# 文件名: colbert/infra/config/base_config.py

import os
import ujson
import dataclasses

# 导入工具函数，用于从 .dnn 文件加载检查点
from colbert.utils.utils import torch_load_dnn
# 导入元数据工具和核心配置类
from utility.utils.save_metadata import get_metadata_only
from .core_config import CoreConfig


@dataclass
class BaseConfig(CoreConfig):
    """
    继承自 CoreConfig，提供了更高级的配置管理功能，特别是
    从不同的来源（如其他配置对象、文件、模型检查点）加载配置。
    """

    @classmethod
    def from_existing(cls, *sources):
        """
        通过合并一个或多个已存在的配置对象来创建一个新的配置对象。
        后面的配置源会覆盖前面配置源中的同名设置。
        """
        kw_args = {}
        for source in sources:
            if source is None:
                continue
            # 只合并源对象中被显式赋值过的字段
            local_kw_args = {k: v for k, v in dataclasses.asdict(source).items() if k in source.assigned}
            kw_args.update(local_kw_args)
        return cls(**kw_args)

    @classmethod
    def from_deprecated_args(cls, args):
        """为了向后兼容，从旧的 argparse 风格的参数对象加载配置。"""
        obj = cls()
        ignored = obj.configure(ignore_unrecognized=True, **args)
        return obj, ignored

    @classmethod
    def from_path(cls, name):
        """从一个 JSON 文件加载配置。"""
        with open(name) as f:
            args = ujson.load(f)
            # 兼容嵌套的 'config' 键
            if 'config' in args:
                args = args['config']
        return cls.from_deprecated_args(args)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """
        从一个模型检查点目录或文件加载配置。
        它会尝试查找 'artifact.metadata' 文件或解析 .dnn 文件。
        """
        if checkpoint_path.endswith('.dnn'):
            dnn = torch_load_dnn(checkpoint_path)
            config, _ = cls.from_deprecated_args(dnn.get('arguments', {}))
            config.set('checkpoint', checkpoint_path)
            return config

        loaded_config_path = os.path.join(checkpoint_path, 'artifact.metadata')
        if os.path.exists(loaded_config_path):
            loaded_config, _ = cls.from_path(loaded_config_path)
            loaded_config.set('checkpoint', checkpoint_path)
            return loaded_config
        
        # 如果检查点路径只是一个 HuggingFace 模型名称 (例如 'bert-base-uncased')，则返回 None
        return None

    @classmethod
    def load_from_index(cls, index_path):
        """从一个 ColBERT 索引目录加载配置。"""
        try:
            # 优先加载 metadata.json
            metadata_path = os.path.join(index_path, 'metadata.json')
            loaded_config, _ = cls.from_path(metadata_path)
        except FileNotFoundError:
            # 兼容旧的 plan.json
            metadata_path = os.path.join(index_path, 'plan.json')
            loaded_config, _ = cls.from_path(metadata_path)
        
        return loaded_config

    def save(self, path, overwrite=False):
        """将当前配置保存到 JSON 文件。"""
        assert overwrite or not os.path.exists(path), f"配置文件 {path} 已存在。"
        with open(path, 'w') as f:
            args = self.export()
            args['meta'] = get_metadata_only()
            args['meta']['version'] = 'colbert-v0.4'
            f.write(ujson.dumps(args, indent=4) + '\n')

    def save_for_checkpoint(self, checkpoint_path):
        """将配置保存为检查点的一部分，文件名为 artifact.metadata。"""
        assert not checkpoint_path.endswith('.dnn'), "不支持保存为旧的 .dnn 格式。"
        output_config_path = os.path.join(checkpoint_path, 'artifact.metadata')
        self.save(output_config_path, overwrite=True)