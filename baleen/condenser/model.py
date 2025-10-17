# 文件名: baleen/condenser/model.py

import torch
import torch.nn as nn
from transformers import ElectraPreTrainedModel, ElectraModel


class ElectraReader(ElectraPreTrainedModel):
    """
    一个基于 Electra 的阅读器模型，用于 Condenser 组件。

    这个模型接收一个拼接了查询和段落的序列，并对序列中的每个 token
    输出一个分数。在 Baleen 的场景中，它被用来预测段落中的哪些句子
    （或 token）与查询最相关，从而实现信息的“冷凝”。
    """
    def __init__(self, config, learn_labels=False):
        super(ElectraReader, self).__init__(config)
        self.electra = ElectraModel(config)
        
        # 一个线性层，用于将 Electra 的隐藏状态映射到分数
        self.linear = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, encoding):
        """
        前向传播。

        Args:
            encoding (dict): HuggingFace Tokenizer 的输出，包含 input_ids, attention_mask 等。

        Returns:
            torch.Tensor: 一个分数张量，形状与输入序列相同。
        """
        # 获取所有 token 的上下文表示
        outputs = self.electra(
            encoding.input_ids,
            attention_mask=encoding.attention_mask,
            token_type_ids=encoding.token_type_ids
        )[0]

        # 计算每个 token 的分数
        scores = self.linear(outputs).squeeze(-1)
        
        # 通过掩码，只保留 [MASK] token 位置的分数，这些位置代表了候选的句子或信息片段
        candidates_mask = (encoding.input_ids == 103) # 103 是 [MASK] 的 token ID
        scores = scores.masked_fill(~candidates_mask, -torch.inf) # 将非候选位置的分数设为负无穷

        return scores