# 文件名: colbert/modeling/reranker/electra.py

import torch.nn as nn
from transformers import ElectraPreTrainedModel, ElectraModel, AutoTokenizer


class ElectraReranker(ElectraPreTrainedModel):
    """
    一个基于 ELECTRA 模型的重排序器（Reranker）。

    与 ColBERT 的“双塔”结构不同，重排序器是一个“交叉注意力”（cross-attention）
    模型。它将查询和文档拼接在一起作为单个输入序列，然后通过 ELECTRA 模型
    来深度融合它们的信息，并最终在 [CLS] token 的位置输出一个相关性分数。

    这种模式通常能达到更高的精度，但计算成本也远高于 ColBERT 的晚期交互。
    """
    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
        # 一个线性层，用于将 [CLS] token 的输出映射到一个单一的相关性分数
        self.linear = nn.Linear(config.hidden_size, 1)
        # 加载对应的分词器
        self.raw_tokenizer = AutoTokenizer.from_pretrained('google/electra-large-discriminator')

        self.init_weights()

    def forward(self, encoding):
        """
        前向传播。

        Args:
            encoding (dict): HuggingFace Tokenizer 的输出，包含 input_ids, attention_mask 等。
        
        Returns:
            torch.Tensor: 一维张量，包含每个 (查询, 文档) 对的相关性分数。
        """
        # 通过 ELECTRA 模型获取所有 token 的上下文表示
        outputs = self.electra(
            encoding.input_ids,
            attention_mask=encoding.attention_mask,
            token_type_ids=encoding.token_type_ids
        )[0]

        # 仅使用 [CLS] token (位于序列的第一个位置) 的输出进行评分
        scores = self.linear(outputs[:, 0]).squeeze(-1)

        return scores
    
    def save(self, path):
        """将模型和分词器保存到磁盘。"""
        assert not path.endswith('.dnn'), f"{path}: 不支持保存为旧的 .dnn 格式。"
        self.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)