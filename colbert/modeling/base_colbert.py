# 文件名: colbert/modeling/base_colbert.py

import torch
from transformers import AutoTokenizer

from colbert.modeling.hf_colbert import HF_ColBERT
from colbert.infra.config import ColBERTConfig


class BaseColBERT(torch.nn.Module):
    """
    一个基础的、浅层的 ColBERT 模块封装。

    这个类将 ColBERT 的核心参数、自定义配置以及底层的 HuggingFace Tokenizer 包装在一起。
    它提供了直接实例化、加载和保存 ColBERT 模型所需的所有组件（模型权重、配置、分词器）的便捷方法。

    默认情况下，模型处于评估模式 (eval mode)。
    """

    def __init__(self, name, colbert_config=None):
        """
        初始化 BaseColBERT。

        Args:
            name (str): 预训练模型的名称（例如 'bert-base-uncased'）或本地模型路径。
            colbert_config (ColBERTConfig, optional): 一个 ColBERTConfig 对象，用于覆盖默认或从检查点加载的配置。
        """
        super().__init__()

        self.name = name
        
        # 合并配置：从检查点加载的配置 -> 传入的 colbert_config
        self.colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(name), colbert_config)
        
        # 从预训练模型加载 ColBERT 的 HuggingFace 实现
        self.model = HF_ColBERT.from_pretrained(name, colbert_config=self.colbert_config)
        
        # 加载与底层 BERT 模型相对应的原始分词器
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.model.base)

        # 默认设置为评估模式
        self.eval()

    @property
    def device(self):
        """返回模型所在的设备 (例如 'cpu' 或 'cuda:0')。"""
        return self.model.device

    @property
    def bert(self):
        """返回底层的 BERT 模型部分。"""
        return self.model.bert

    @property
    def linear(self):
        """返回用于降维的线性层。"""
        return self.model.linear

    def save(self, path):
        """
        将完整的 ColBERT 模型（权重、分词器、配置）保存到指定路径。
        这使用了 HuggingFace Transformers 的 `save_pretrained` 标准格式。

        Args:
            path (str): 保存模型的目录路径。
        """
        assert not path.endswith('.dnn'), f"{path}: .dnn 是旧的检查点格式，不再支持保存为此格式。"

        # 保存模型权重
        self.model.save_pretrained(path)
        # 保存分词器
        self.raw_tokenizer.save_pretrained(path)
        # 保存 ColBERT 的特定配置
        self.colbert_config.save_for_checkpoint(path)