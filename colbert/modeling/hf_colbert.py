# 文件名: colbert/modeling/hf_colbert.py

import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer

# 导入工具函数，用于从旧的 .dnn 格式加载检查点
from colbert.utils.utils import torch_load_dnn


class HF_ColBERT(BertPreTrainedModel):
    """
    一个基于 HuggingFace Transformers `BertPreTrainedModel` 的浅层包装类。

    这个类定义了 ColBERT 模型的核心架构：一个标准的 BERT 模型后面跟着一个
    用于降维的线性层。

    通过继承 `BertPreTrainedModel`，我们可以方便地利用 HuggingFace 生态系统
    的功能，例如 `from_pretrained` 和 `save_pretrained`。
    """
    # 当加载模型时，如果遇到这些名称的权重，则忽略（因为它们是 reranker 特有的）
    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config, colbert_config):
        """
        初始化 HF_ColBERT 模型。

        Args:
            config (BertConfig): HuggingFace 的 BERT 配置对象。
            colbert_config (ColBERTConfig): ColBERT 的特定配置对象。
        """
        super().__init__(config)

        # ColBERT 的输出维度
        self.dim = colbert_config.dim
        # 基础的 BERT 模型
        self.bert = BertModel(config)
        # 用于将 BERT 的隐藏状态降维到 self.dim 的线性层
        self.linear = nn.Linear(config.hidden_size, colbert_config.dim, bias=False)

        # 初始化新增的线性层的权重
        self.init_weights()

    @classmethod
    def from_pretrained(cls, name_or_path, colbert_config):
        """
        重写 `from_pretrained` 方法，以支持从 ColBERT 的自定义检查点格式加载。

        它能处理两种情况：
        1.  旧的 .dnn 格式检查点。
        2.  标准的 HuggingFace 模型目录。
        """
        # 处理旧的 .dnn 格式
        if name_or_path.endswith('.dnn'):
            dnn = torch_load_dnn(name_or_path)
            # 从 .dnn 文件中提取基础 BERT 模型的名称 (例如 'bert-base-uncased')
            base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')
            
            # 使用 BertPreTrainedModel 的 from_pretrained 加载模型，并传入提取的状态字典
            obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], colbert_config=colbert_config)
            obj.base = base
            return obj

        # 处理标准的 HuggingFace 模型名称或路径
        obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)
        obj.base = name_or_path
        return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path):
        """
        一个静态方法，用于从模型名称或路径加载对应的原始 HuggingFace 分词器。
        同样支持 .dnn 格式。
        """
        if name_or_path.endswith('.dnn'):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')
            obj = AutoTokenizer.from_pretrained(base)
            obj.base = base
            return obj

        obj = AutoTokenizer.from_pretrained(name_or_path)
        obj.base = name_or_path
        return obj