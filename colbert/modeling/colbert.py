# 文件名: colbert/modeling/colbert.py

import torch
import string

from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.modeling.base_colbert import BaseColBERT


class ColBERT(BaseColBERT):
    """
    ColBERT 模型的核心实现。

    这个类处理 ColBERT 中基本的编码和评分操作，主要用于模型训练。
    它定义了 `forward` 方法，该方法接收查询和文档的 token ID，计算它们之间的
    相关性分数，并返回用于优化的损失。
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        """
        初始化 ColBERT 模型。

        Args:
            name (str, optional): 预训练模型的名称或路径。
            colbert_config (ColBERTConfig, optional): 自定义配置对象。
        """
        super().__init__(name, colbert_config)

        # 如果配置中启用了标点符号过滤，则创建一个 skiplist
        if self.colbert_config.mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}

    def forward(self, Q, D):
        """
        模型的前向传播函数，用于训练。

        Args:
            Q (tuple[torch.Tensor, torch.Tensor]): 查询的 (input_ids, attention_mask)。
            D (tuple[torch.Tensor, torch.Tensor]): 文档的 (input_ids, attention_mask)。

        Returns:
            torch.Tensor or tuple:
                - scores: 每个查询与其对应正负样本对之间的相似度分数。
                - ib_loss (optional): 如果启用了 in-batch negatives, 则额外返回该损失项。
        """
        # 1. 编码查询
        Q_embs = self.query(*Q)
        # 2. 编码文档
        D_embs, D_mask = self.doc(*D, keep_dims='return_mask')

        # 3. 计算分数：将每个查询嵌入复制 n-way 次，以匹配对应的 n 个文档
        if self.colbert_config.nway > 1:
            Q_embs = Q_embs.repeat_interleave(self.colbert_config.nway, dim=0)
        
        scores = self.score(Q_embs, D_embs, D_mask)

        # 4. (可选) 计算 in-batch negatives 损失
        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q_embs, D_embs, D_mask)
            return scores, ib_loss

        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        """
        计算 in-batch negatives 损失。
        每个批次内的其他文档都可以被视为当前查询的负样本。
        """
        # 计算所有查询与所有文档之间的相似度矩阵
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)
        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)

        # ... (此处省略了构建 in-batch negative 标签和计算损失的复杂逻辑) ...
        # 核心思想是，对于每个查询，其正样本是对应的文档，而批次内所有其他文档都是负样本
        
        # 此处简化逻辑，实际代码更复杂
        labels = torch.arange(Q.size(0), device=scores.device) * self.colbert_config.nway
        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask):
        """
        将查询的 token ID 编码为嵌入矩阵。
        """
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        # 1. 通过 BERT 获取 token 的上下文表示
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # 2. 通过线性层进行降维
        Q = self.linear(Q)
        # 3. 对特殊 token ([CLS], [SEP], [Q]) 进行掩码 (masking)
        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask
        # 4. L2 归一化
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        """
        将文档的 token ID 编码为嵌入矩阵。
        """
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        
        # 掩码特殊 token 和标点符号
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        # 归一化并转换为半精度以节省内存
        D = torch.nn.functional.normalize(D, p=2, dim=2).half()

        if keep_dims == 'return_mask':
            return D, mask.bool()
        elif not keep_dims:
             # 返回变长嵌入列表
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            return [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D_padded, D_mask):
        """
        计算查询和文档之间的相关性分数。
        """
        # 根据配置选择相似度计算方式
        if self.colbert_config.similarity == 'l2':
            # L2 距离的负值
            return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
        
        # 默认使用余弦相似度 (通过 colbert_score 实现)
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        """
        生成一个布尔掩码，用于在计算中忽略特定的 token。
        """
        mask = [[(token_id not in skiplist) and (token_id != 0) for token_id in d] for d in input_ids.cpu().tolist()]
        return mask


def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
    ColBERT 评分函数的核心 (MaxSim 操作)。

    1.  计算查询中每个词元与文档中所有词元之间的相似度（点积）。
    2.  对于文档中的每个词元，找到与之最相似的查询词元，记录下这个最大相似度。
    3.  将文档中所有词元的最大相似度得分相加，得到最终的相关性分数。
    """
    Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()
    
    # 计算点积相似度矩阵
    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    """对相似度矩阵进行规约（reduction）操作，得到最终分数。"""
    # 1. 将填充部分的得分设为一个很小的负数，以在 max 操作中忽略它们
    padding_mask = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[padding_mask] = -9999
    
    # 2. 对每个文档词元，取其与所有查询词元相似度的最大值
    scores = scores_padded.max(1).values
    
    # 3. 将所有最大值相加
    return scores.sum(-1)

# (colbert_score_packed 函数为优化版本，此处省略注释以保持简洁)