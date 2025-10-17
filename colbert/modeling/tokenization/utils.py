# 文件名: colbert/modeling/tokenization/utils.py

import torch


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, passages, scores, bsize, nway):
    """
    将三元组（或 n-way）数据转换为模型训练所需的张量格式。

    它接收原始文本列表，使用相应的分词器进行处理，并将它们组织成
    适合输入到 ColBERT 模型 `forward` 方法的批次。

    Args:
        query_tokenizer (QueryTokenizer): 查询分词器。
        doc_tokenizer (DocTokenizer): 文档分词器。
        queries (list[str]): 查询文本列表。
        passages (list[str]): 段落文本列表，其长度应为 `len(queries) * nway`。
        scores (list[float]): 与每个段落相关的分数（用于知识蒸馏）。
        bsize (int): 批处理大小。
        nway (int): 每个查询对应的段落数量。

    Returns:
        list[tuple]: 一个包含多个批次的列表，每个批次是一个元组，
                     通常为 ((Q_ids, Q_mask), (D_ids, D_mask), batch_scores)。
    """
    # 1. 对查询和段落进行分词和张量化
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(passages)

    # 2. 将张量划分为多个批次
    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    doc_batches = _split_into_batches(D_ids, D_mask, bsize * nway)
    
    # 3. 如果提供了分数，也对分数进行分批
    if len(scores):
        score_batches = _split_into_batches2(scores, bsize * nway)
    else:
        score_batches = [[] for _ in doc_batches]

    # 4. 将查询、文档和分数组合成最终的批次列表
    batches = []
    for Q, D, S in zip(query_batches, doc_batches, score_batches):
        batches.append((Q, D, S))

    return batches


def _sort_by_length(ids, mask, bsize):
    """
    一个辅助函数，用于根据序列的实际长度（由 mask 计算）对一个批次进行排序。
    在处理变长序列时，将长度相近的序列放在一起可以提高计算效率（尤其是在 RNN 中，
    但在 Transformer 中也有一定好处，可以减少填充量）。

    Returns:
        tuple: (排序后的 ids, 排序后的 mask, 用于恢复原始顺序的索引)
    """
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))
    
    # 计算每个序列的长度
    lengths = mask.sum(-1)
    # 获取排序后的索引
    indices = lengths.sort().indices
    # 获取用于恢复原始顺序的“逆索引”
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    """一个简单的辅助函数，将大的张量切分为指定大小的批次。"""
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))
    return batches


def _split_into_batches2(scores, bsize):
    """一个简单的辅助函数，将列表切分为指定大小的批次。"""
    batches = []
    for offset in range(0, len(scores), bsize):
        batches.append(scores[offset:offset+bsize])
    return batches