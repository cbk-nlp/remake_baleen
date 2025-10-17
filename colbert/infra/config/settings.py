# 文件名: colbert/infra/config/settings.py

import os
import torch
import __main__
from dataclasses import dataclass
from colbert.utils.utils import timestamp
from .core_config import DefaultVal


@dataclass
class RunSettings:
    """定义与实验运行环境相关的设置。"""
    overwrite: bool = DefaultVal(False)
    root: str = DefaultVal(os.path.join(os.getcwd(), 'experiments'))
    experiment: str = DefaultVal('default')
    index_root: str = DefaultVal(None)
    name: str = DefaultVal(timestamp(daydir=True))
    rank: int = DefaultVal(0)
    nranks: int = DefaultVal(1)
    amp: bool = DefaultVal(True)
    gpus: int = DefaultVal(torch.cuda.device_count())

@dataclass
class ResourceSettings:
    """定义与数据和模型资源路径相关的设置。"""
    checkpoint: str = DefaultVal(None)
    triples: str = DefaultVal(None)
    collection: str = DefaultVal(None)
    queries: str = DefaultVal(None)
    index_name: str = DefaultVal(None)

@dataclass
class DocSettings:
    """定义与文档编码相关的设置。"""
    dim: int = DefaultVal(128)
    doc_maxlen: int = DefaultVal(220)
    mask_punctuation: bool = DefaultVal(True)

@dataclass
class QuerySettings:
    """定义与查询编码相关的设置。"""
    query_maxlen: int = DefaultVal(32)
    attend_to_mask_tokens: bool = DefaultVal(False)
    interaction: str = DefaultVal('colbert')

@dataclass
class TrainingSettings:
    """定义与模型训练相关的超参数。"""
    similarity: str = DefaultVal('cosine')
    bsize: int = DefaultVal(32)
    accumsteps: int = DefaultVal(1)
    lr: float = DefaultVal(3e-06)
    maxsteps: int = DefaultVal(500_000)
    save_every: int = DefaultVal(None)
    resume: bool = DefaultVal(False)
    warmup: int = DefaultVal(None)
    warmup_bert: int = DefaultVal(None)
    relu: bool = DefaultVal(False)
    nway: int = DefaultVal(2)
    use_ib_negatives: bool = DefaultVal(False)
    reranker: bool = DefaultVal(False)
    distillation_alpha: float = DefaultVal(1.0)
    ignore_scores: bool = DefaultVal(False)

@dataclass
class IndexingSettings:
    """定义与索引构建相关的设置。"""
    index_path: str = DefaultVal(None)
    nbits: int = DefaultVal(1)
    kmeans_niters: int = DefaultVal(20)
    
    @property
    def index_path_(self):
        # 如果 index_path 未指定，则根据 index_root 和 index_name 自动生成
        return self.index_path or os.path.join(self.index_root_, self.index_name)

@dataclass
class SearchSettings:
    """定义与搜索/检索相关的设置。"""
    nprobe: int = DefaultVal(2)
    ncandidates: int = DefaultVal(8192)