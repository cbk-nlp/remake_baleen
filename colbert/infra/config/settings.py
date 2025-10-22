# 文件名: colbert/infra/config/settings.py

import os
import torch

import __main__
from dataclasses import dataclass
from colbert.utils.utils import timestamp

from .core_config import DefaultVal


@dataclass
class RunSettings:
    """
    定义与单次“运行”（Run）相关的环境和基本设置。
    这里的默认值在 Run() 中具有特殊状态，它们是初始化的硬默认值。
    """

    # 是否覆盖已有的实验/索引/日志
    overwrite: bool = DefaultVal(False)

    # 实验的根目录，默认为当前工作目录下的 'experiments'
    root: str = DefaultVal(os.path.join(os.getcwd(), 'experiments'))
    # 实验的名称，用于组织相关的多次运行
    experiment: str = DefaultVal('default')

    # 索引的根目录。如果为 None，将从 'root' 和 'experiment' 派生
    index_root: str = DefaultVal(None)
    # 本次运行的唯一名称，默认为当前时间戳（例如 '2025-10-20/14.30.00'）
    name: str = DefaultVal(timestamp(daydir=True))

    # 当前进程在分布式环境中的排名（ID），默认为 0 (主进程)
    rank: int = DefaultVal(0)
    # 分布式环境中的总进程数，默认为 1 (非分布式)
    nranks: int = DefaultVal(1)
    # 是否使用自动混合精度 (Automatic Mixed Precision, AMP)
    amp: bool = DefaultVal(True)

    # (类属性) 获取系统中可见的 GPU 总数
    total_visible_gpus = torch.cuda.device_count()
    # 要使用的 GPU 数量（int）或 GPU ID 列表（str, e.g., '0,2'）。
    # 默认为所有可见的 GPU。
    gpus: int = DefaultVal(total_visible_gpus)

    @property
    def gpus_(self):
        """
        一个属性，用于解析 'gpus' 字段，将其标准化为一个排序后的 GPU ID 列表。

        例如:
        - gpus=2 -> [0, 1]
        - gpus='0,3,1' -> [0, 1, 3]
        
        返回:
            list[int]: GPU ID 列表。
        """
        value = self.gpus

        if isinstance(value, int):
            # 如果 gpus 是一个整数 (例如 4)，则将其转换为 [0, 1, 2, 3]
            value = list(range(value))

        if isinstance(value, str):
            # 如果 gpus 是一个字符串 (例如 '0,2,3')，则将其分割
            value = value.split(',')

        # 转换为整数列表，去重并排序
        value = list(map(int, value))
        value = sorted(list(set(value)))

        # 确保所有指定的 GPU ID 都是有效的
        assert all(device_idx in range(0, self.total_visible_gpus) for device_idx in value), value

        return value

    @property
    def index_root_(self):
        """
        一个属性，用于获取索引的根目录。
        如果 'index_root' (self.index_root) 未设置，
        则自动生成一个默认路径 (e.g., /path/to/root/experiment/indexes/)。
        """
        return self.index_root or os.path.join(self.root, self.experiment, 'indexes/')

    @property
    def script_name_(self):
        """
        一个属性，用于自动检测并生成运行此脚本的 Python 文件的名称。
        它会将文件路径转换为模块路径格式 (例如 'a/b/c.py' -> 'a.b.c')。
        如果无法检测（例如在 REPL 中），则返回 'none'。
        """
        if '__file__' in dir(__main__):
            cwd = os.path.abspath(os.getcwd())
            script_path = os.path.abspath(__main__.__file__)
            root_path = os.path.abspath(self.root)

            if script_path.startswith(cwd):
                # 如果脚本在当前工作目录下，移除CWD前缀
                script_path = script_path[len(cwd):]
            
            else:
                try:
                    # 否则，尝试找到脚本和 root 目录的公共路径并移除
                    commonpath = os.path.commonpath([script_path, root_path])
                    script_path = script_path[len(commonpath):]
                except:
                    pass
            
            # 将文件路径 (a/b/c.py) 转换为模块名 (a.b.c)
            assert script_path.endswith('.py')
            script_name = script_path.replace('/', '.').strip('.')[:-3]

            assert len(script_name) > 0, (script_name, script_path, cwd)
            
            return script_name
        
        # 如果不是作为脚本运行（例如在 REPL 中），则返回 'none'
        return 'none'

    @property
    def path_(self):
        """
        一个属性，用于生成本次运行的完整、唯一的输出路径。
        (e.g., /path/to/root/experiment/script_name/run_name)
        """
        return os.path.join(self.root, self.experiment, self.script_name_, self.name)

    @property
    def device_(self):
        """
        一个属性，根据当前进程的 'rank' 从 'gpus_' 列表中为其分配一个特定的 GPU 设备 ID。
        (例如，rank 2, gpus_=[0, 1, 4, 7] -> device 4)
        """
        # 使用取模运算(%)来实现循环分配
        return self.gpus_[self.rank % len(self.gpus_)]


@dataclass
class ResourceSettings:
    """
    定义模型运行所需的各种资源文件路径。
    """
    # 预训练模型检查点路径 (用于加载模型权重)
    checkpoint: str = DefaultVal(None)
    # 训练用的三元组文件 (query, positive, negative) 路径
    triples: str = DefaultVal(None)
    # 文档集合文件路径 (e.g., collection.tsv)
    collection: str = DefaultVal(None)
    # 查询集文件路径 (e.g., queries.tsv)
    queries: str = DefaultVal(None)
    # 要加载或创建的索引的名称
    index_name: str = DefaultVal(None)


@dataclass
class DocSettings:
    """
    与文档（passage）处理相关的设置。
    """
    # ColBERT 嵌入的维度 (例如 128)
    dim: int = DefaultVal(128)
    # 文档的最大长度（以 token 计）。超过此长度的将被截断。
    doc_maxlen: int = DefaultVal(220)
    # 在编码文档时是否忽略（掩码）标点符号
    mask_punctuation: bool = DefaultVal(True)


@dataclass
class QuerySettings:
    """
    与查询（query）处理相关的设置。
    """
    # 查询的最大长度（以 token 计）
    query_maxlen: int = DefaultVal(32)
    # (BERT特定) 是否允许模型在注意力机制中关注 [MASK] 标记
    attend_to_mask_tokens : bool = DefaultVal(False)
    # 交互类型，例如 'colbert' (默认的 MaxSim) 或 'flipr'
    interaction: str = DefaultVal('colbert')


@dataclass
class TrainingSettings:
    """
    与模型训练过程相关的设置。
    """
    # 用于计算嵌入相似度的度量，例如 'cosine' (余弦) 或 'l2' (欧氏距离)
    similarity: str = DefaultVal('cosine')

    # 批次大小 (Batch size)
    bsize: int = DefaultVal(32)

    # 梯度累积的步数。有效批次大小 = bsize * accumsteps
    accumsteps: int = DefaultVal(1)

    # 学习率
    lr: float = DefaultVal(3e-06)

    # 最大训练步数
    maxsteps: int = DefaultVal(500_000)

    # 每隔多少步保存一次检查点。如果为 None，则可能按 epoch 保存。
    save_every: int = DefaultVal(None)

    # 是否从最新的检查点恢复训练
    resume: bool = DefaultVal(False)

    ## NEW (新添加的训练设置):
    # 学习率预热（warm-up）的步数
    warmup: int = DefaultVal(None)
    
    # (可选) 仅预热 BERT 部分的步数
    warmup_bert: int = DefaultVal(None)

    # 是否在 ColBERT 线性层后使用 ReLU 激活
    relu: bool = DefaultVal(False)

    # 对比损失 (contrastive loss) 中的 "n-way" 设置。
    # (总数 = 1个正例 + (n-1)个负例)
    nway: int = DefaultVal(2)

    # 是否使用批内负采样 (In-Batch Negatives) 作为额外的负例
    use_ib_negatives: bool = DefaultVal(False)

    # 是否作为重排器 (reranker) 模式进行训练
    reranker: bool = DefaultVal(False)

    # (用于知识蒸馏) 蒸馏损失的 alpha 权重
    distillation_alpha: float = DefaultVal(1.0)

    # (用于知识蒸馏) 是否忽略教师模型的得分
    ignore_scores: bool = DefaultVal(False)


@dataclass
class IndexingSettings:
    """
    与构建（或加载）索引相关的设置。
    """
    # 索引的完整路径。如果为 None，将自动生成。
    index_path: str = DefaultVal(None)

    # 用于量化（如 PQ）的位数。例如 1, 2, 4, 8。
    # nbits=1 对应于二值化 (binary quantization)。
    nbits: int = DefaultVal(1)

    # K-Means 聚类（用于量化码本）的迭代次数
    kmeans_niters: int = DefaultVal(20)
    
    @property
    def index_path_(self):
        """
        一个属性，用于获取索引的完整路径。
        如果 'index_path' 未设置，则使用 'index_root_' 和 'index_name' 自动生成。
        (e.g., /path/to/indexes/my_index_name)
        """
        return self.index_path or os.path.join(self.index_root_, self.index_name)

@dataclass
class SearchSettings:
    """
    与搜索（检索）过程相关的设置。
    """
    # (用于 FAISS 等索引) 在搜索时要探查的聚类（簇）的数量。
    # nprobe 越大，召回率越高，但速度越慢。
    nprobe: int = DefaultVal(2)
    
    # (用于 ANN 近似最近邻) 在最终重排前要检索的候选文档数量。
    ncandidates: int = DefaultVal(8192)