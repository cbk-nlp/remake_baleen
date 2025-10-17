# 文件名: colbert/infra/config/config.py

from dataclasses import dataclass
from .base_config import BaseConfig
from .settings import *


@dataclass
class RunConfig(BaseConfig, RunSettings):
    """
    一个专门用于管理实验运行环境配置的类。
    它继承了 BaseConfig 的所有功能和 RunSettings 的所有字段。
    """
    pass


@dataclass
class ColBERTConfig(RunSettings, ResourceSettings, DocSettings, QuerySettings, TrainingSettings,
                    IndexingSettings, SearchSettings, BaseConfig):
    """
    ColBERT 的主配置类。

    它通过多重继承，将所有不同类别的设置 (Run, Resource, Doc, Query, Training,
    Indexing, Search) 组合到一个单一、全面的配置对象中。
    同时，它也继承了 BaseConfig 的高级配置管理能力。
    """
    pass