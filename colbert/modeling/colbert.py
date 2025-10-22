# 文件名: colbert/modeling/colbert.py

from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor # 导入StridedTensor，用于高效处理变长的“打包”张量
from colbert.utils.utils import print_message, flatten
from colbert.modeling.base_colbert import BaseColBERT

import torch
import string


class ColBERT(BaseColBERT):
    """
    此类处理 ColBERT 中的基本编码和评分操作。它主要用于训练阶段。
    (ColBERT (Contextualized Late Interaction over BERT) 是一种用于信息检索的
    深度学习模型，它独立编码查询和文档，然后在嵌入层面上进行“延迟交互” (Late Interaction)。)
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        """
        初始化 ColBERT 模型。

        参数:
            name (str): 
                基础 BERT 模型的名称 (例如 'bert-base-uncased')，
                将从 Hugging Face Hub 加载。
            colbert_config (ColBERTConfig, optional): 
                ColBERT 的特定配置对象。如果为 None，将使用默认配置。
        """
        super().__init__(name, colbert_config)

        if self.colbert_config.mask_punctuation:
            # 如果配置要求在编码文档时“掩码”(忽略)标点符号，
            # 则创建一个“跳过列表” (skiplist)。
            # 这个列表包含所有标点符号字符及其在分词器中的对应 token ID。
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}

    def forward(self, Q, D):
        """
        模型的前向传播函数，主要用于训练。
        它接收一批查询(Q)和文档(D)，计算它们之间的相关性得分。

        参数:
            Q (tuple): 包含查询 'input_ids' 和 'attention_mask' 的元组。
            D (tuple): 包含文档 'input_ids' 和 'attention_mask' 的元组。

        返回:
            torch.Tensor: 批次中每个查询-文档对的得分 (B)。
            torch.Tensor (optional): 
                如果 use_ib_negatives (使用批内负采样) 为 True，
                则额外返回批内负采样(in-batch negatives)的损失。
        """
        # 1. 分别编码查询和文档
        # Q shape: (B_q, Q_len, dim)
        Q = self.query(*Q)
        # D shape: (B_d, D_len, dim), D_mask shape: (B_d, D_len, 1)
        D, D_mask = self.doc(*D, keep_dims='return_mask')

        # 在训练中 (n-way contrastive loss)，每个查询 Q 
        # 需要与其对应的 n-way 个文档 D 进行比较。
        # (通常 B_d = B_q * nway, 其中 nway=1个正例 + (nway-1)个负例)
        # 因此，我们将每个 Q 向量重复 nway 次，使其与 D 批次对齐。
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        
        # 2. 计算对齐的 Q-D 得分 (MaxSim)
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            # 3. (可选) 计算批内负采样 (In-Batch Negatives) 损失
            #    这会计算批次中每个 Q 与 *所有* D 之间的得分，
            #    并将非对应的 D 作为额外负例。
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss

        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        """
        计算批内负采样 (In-Batch Negatives) 的损失。
        这是一种对比学习技术，其中批次内的其他样本被用作负例。

        参数:
            Q (Tensor): 查询嵌入 (B_q, Q_len, dim)。
            D (Tensor): 文档嵌入 (B_d, D_len, dim)。B_d = B_q * nway。
            D_mask (Tensor): 文档掩码 (B_d, D_len, 1)。

        返回:
            torch.Tensor: IB 损失 (一个标量)。
        """
        # 1. 计算所有 Q 与所有 D 之间的得分
        # Q (B_q, Q_len, dim) -> Q.permute (B_q, dim, Q_len)
        # D (B_d, D_len, dim)
        # 目标: (B_q, B_d, Q_len, D_len)
        # (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1))
        #   (1, B_d, D_len, dim) @ (B_q, 1, dim, Q_len)
        #   -> (B_q, B_d, D_len, Q_len)
        # .flatten(0, 1) -> (B_q * B_d, D_len, Q_len)
        # .permute(0, 2, 1) -> (B_q * B_d, Q_len, D_len) [与 colbert_score_reduce 格式一致]
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).permute(0, 1, 3, 2).flatten(0, 1)

        # 2. Reduce: (B_q * B_d, Q_len, D_len) -> (B_q * B_d)
        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)

        # 3. 筛选负例
        nway = self.colbert_config.nway
        # `all_except_self_negatives` 逻辑是：对于第 qidx 个查询，
        # 我们选取所有 (qidx, d) 的得分，
        # *除了* 属于 qidx 自己的 n-way 文档组 (即 d 在 [qidx*nway, (qidx+1)*nway) 范围内的)。
        # (注意：代码实现似乎是选取了 *除了* 对应正例之外的所有样本)
        all_except_self_negatives = [list(range(qidx*D.size(0), qidx*D.size(0) + nway*qidx+1)) +
                                     list(range(qidx*D.size(0) + nway * (qidx+1), qidx*D.size(0) + D.size(0)))
                                     for qidx in range(Q.size(0))]

        scores = scores[flatten(all_except_self_negatives)]
        scores = scores.view(Q.size(0), -1) # (B_q, B_d - nway + 1 ?)

        # 4. 计算交叉熵损失
        # 对应的“正例”标签（在 n-way 组中的第一个）
        labels = torch.arange(0, Q.size(0), device=scores.device) * (self.colbert_config.nway)
        
        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask):
        """
        将查询的 input_ids 编码为 ColBERT 嵌入 (Q 向量)。

        参数:
            input_ids (Tensor): 查询 token IDs (B, Q_len)。
            attention_mask (Tensor): 查询的注意力掩码 (B, Q_len)。

        返回:
            torch.Tensor: 归一化后的查询嵌入 (B, Q_len, dim)。
        """
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        
        # 1. 通过基础 BERT 模型
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        # 2. 通过线性层 (通常是降维)
        Q = self.linear(Q)

        # 3. 应用掩码
        #    查询的掩码 (skiplist=[]) 只去除 [CLS], [SEP] 和 [PAD] (id=0)
        #    它 *保留* 标点符号，因为标点在查询中可能很重要。
        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        # 4. L2 归一化 (在最后一个维度上)
        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        """
        将文档的 input_ids 编码为 ColBERT 嵌入 (D 向量)。

        参数:
            input_ids (Tensor): 文档 token IDs (B, D_len)。
            attention_mask (Tensor): 文档的注意力掩码 (B, D_len)。
            keep_dims (bool or str): 维度保留策略。
                True: 返回填充后的 Tensor (B, D_len, dim)。
                False: 返回一个列表 (长度为 B)，每个元素是去除填充和标点后的
                       有效 token 嵌入 (在 CPU 上)。(用于索引构建)
                'return_mask': 返回填充后的 Tensor 和对应的布尔掩码。(用于训练)

        返回:
            torch.Tensor or (torch.Tensor, torch.Tensor) or list: 根据 keep_dims 返回不同内容。
        """
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        
        # 1. 通过基础 BERT 模型
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        
        # 2. 通过线性层 (降维)
        D = self.linear(D)

        # 3. 应用掩码
        #    文档的掩码 (self.skiplist) 会去除 [CLS], [SEP], [PAD] 
        #    *以及* 所有在 __init__ 中定义的标点符号。
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        # 4. L2 归一化 (并转为半精度 half() 以节省内存)
        D = torch.nn.functional.normalize(D, p=2, dim=2).half()

        if keep_dims is False:
            # 用于索引构建：返回一个 CPU 上的列表
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            # 列表推导式：对批次中的每个 D，只保留 mask 为 True 的嵌入
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            # 用于训练：返回 D 嵌入和掩码
            return D, mask.bool()

        # 默认 (keep_dims=True)：返回填充的 D 嵌入
        return D

    def score(self, Q, D_padded, D_mask):
        """
        计算查询(Q)和文档(D)之间的 ColBERT (MaxSim) 得分。
        这是 ColBERT 交互的核心。

        参数:
            Q (Tensor): 查询嵌入 (B, Q_len, dim)。
            D_padded (Tensor): (填充后的) 文档嵌入 (B, D_len, dim)。
            D_mask (Tensor): 文档掩码 (B, D_len, 1)，True 为有效 token。

        返回:
            torch.Tensor: 每个 Q-D 对的最终得分 (B)。
        """
        # assert self.colbert_config.similarity == 'cosine'

        if self.colbert_config.similarity == 'l2':
            # 如果配置为 L2 距离 (平方欧氏距离的负数)
            assert self.colbert_config.interaction == 'colbert'
            # (Q.unsqueeze(2) - D_padded.unsqueeze(1))**2 -> (B, Q_len, D_len, dim)
            # .sum(-1) -> (B, Q_len, D_len)
            # .max(-1).values -> (B, Q_len) (MaxSim over D_len)
            # .sum(-1) -> (B) (Sum over Q_len)
            return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

        # 默认：使用余弦相似度 (通过 colbert_score 实现)
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        """
        一个辅助函数，用于根据 skiplist (跳过列表) 和填充 (id=0) 创建布尔掩码。

        参数:
            input_ids (Tensor): Token IDs (在 CPU 上)。
            skiplist (dict or list): 要跳过 (Mask=False) 的 token IDs。

        返回:
            list: 布尔值的嵌套列表 (B, Len)，表示哪些 token 应该被保留 (True)。
        """
        mask = [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask


# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer

# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    """
    ColBERT 评分的“Reduce”步骤。
    获取一个 (B, Q_len, D_len) 的相似度矩阵，将其规约为 (B) 的最终得分。
    这是 ColBERT "MaxSim" 交互的核心操作。

    参数:
        scores_padded (Tensor): 填充后的 Q-D 相似度矩阵 (B, Q_len, D_len)。
        D_mask (Tensor): 文档掩码 (B, D_len, 1)。
        config (ColBERTConfig): ColBERT 配置。

    返回:
        torch.Tensor: 最终得分 (B)。
    """
    # 1. 应用文档掩码 (D_mask)
    # D_mask 是 (B, D_len, 1)，D_padding 是 (B, D_len)
    # D_padding 为 True 的地方是填充 (padding) 或被掩码的标点
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    
    # 将填充位置的得分设为极小值，使其在 max 操作中被忽略
    # scores_padded[D_padding] = -9999 
    # (注意：这里 D_padding 是 (B, D_len)，而 scores_padded 是 (B, Q_len, D_len))
    # PyTorch 的广播机制会正确处理：(B, 1, D_len)
    scores_padded[D_padding.unsqueeze(1)] = -9999 

    # 2. MaxSim 操作: 沿 D_len 维度取最大值
    #    对于 Q 中的 *每个* token，找到 D 中 *最相似* 的 token。
    # (B, Q_len, D_len) -> (B, Q_len)
    scores = scores_padded.max(2).values

    assert config.interaction in ['colbert', 'flipr'], config.interaction

    if config.interaction == 'flipr':
        # FLIPR 的特殊评分逻辑 (取 Q_len 的前 K1 和后 K2 个 token)
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)

        return A + B

    # 3. 默认 ColBERT: 沿 Q_len 维度求和
    #    将每个查询 token 的最大相似度得分相加。
    # (B, Q_len) -> (B)
    return scores.sum(-1)


# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
    计算 ColBERT 得分 (核心 MaxSim 计算)。
    通过矩阵乘法计算 (Q @ D.T)，然后调用 colbert_score_reduce 函数。

    支持两种模式:
      1. Q (1, Q_len, dim) vs D (B, D_len, dim) -> (B) 
         (一个查询 vs B个文档，例如在检索时)
      2. Q (B, Q_len, dim) vs D (B, D_len, dim) -> (B) 
         (B个查询 vs B个对齐的文档，例如在训练时)

    参数:
        Q (Tensor): 查询嵌入。
        D_padded (Tensor): (填充后的) 文档嵌入。
        D_mask (Tensor): 文档掩码。
        config (ColBERTConfig): ColBERT 配置。

    返回:
        torch.Tensor: 最终得分 (B)。
    """
    # 将所有张量移动到 GPU
    Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)] # 检查是否为上述两种模式之一

    # 核心计算：批量矩阵乘法
    # (B, D_len, dim) @ (B, dim, Q_len) -> (B, D_len, Q_len)
    # 注意：D 是 (B, D_len, dim), Q 是 (B, Q_len, dim)
    # 我们需要 (D @ Q.T) (按 B 批次)
    # Q.to(dtype=D_padded.dtype) 是为了匹配 D 的半精度 (half)
    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    # (B, D_len, Q_len) -> (B, Q_len, D_len) (为了匹配 reduce 函数的输入)
    scores = scores.permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_packed(Q, D_packed, D_lengths, config=ColBERTConfig()):
    """
    计算 ColBERT 得分，但针对的是“打包”(packed)的文档嵌入。
    “打包”意味着所有文档的有效嵌入被连接成一个 (N_tokens, dim) 的大张量，
    以避免填充带来的计算和内存开销。

    此函数仅适用于单个查询 (Q.size(0) == 1)。

    参数:
        Q (Tensor): 单个查询嵌入 (1, Q_len, dim)。
        D_packed (Tensor): 打包的文档嵌入 (N_all_tokens, dim)。
        D_lengths (Tensor): 包含每个文档长度的 1D 张量 (B)。
        config (ColBERTConfig): ColBERT 配置。

    返回:
        torch.Tensor: 每个文档的最终得分 (B)。
    """
    Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0) # (1, Q_len, dim) -> (Q_len, dim)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    # 1. 计算所有 token 对 (N_all_tokens, Q_len)
    #    (N_all_tokens, dim) @ (dim, Q_len)
    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    # 2. 使用 StridedTensor 将打包的 scores 还原为 (B, D_len, Q_len)
    #    D_lengths 告诉 StridedTensor 如何从 (N_all_tokens, Q_len) 中切分出每个文档
    scores_padded, scores_mask = StridedTensor(scores, D_lengths).as_padded_tensor()

    # (B, D_len, Q_len) -> (B, Q_len, D_len)
    scores_padded = scores_padded.permute(0, 2, 1)
    
    # (B, D_len) -> (B, D_len, 1) (适配 reduce 函数的 D_mask 格式)
    scores_mask = scores_mask.unsqueeze(-1)

    # 3. Reduce
    return colbert_score_reduce(scores_padded, scores_mask, config)