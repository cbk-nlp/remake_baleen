# 文件名: colbert/modeling/tokenization/doc_tokenization.py

import torch
from colbert.modeling.hf_colbert import HF_ColBERT
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization.utils import _split_into_batches, _sort_by_length


class DocTokenizer:
    """为 ColBERT 的文档（passages）设计的专用分词器。"""
    def __init__(self, config: ColBERTConfig):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)
        self.config = config
        self.doc_maxlen = config.doc_maxlen

        # 定义文档的特殊标记 [D]
        self.D_marker_token, self.D_marker_token_id = '[D]', self.tok.convert_tokens_to_ids('[unused1]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

        assert self.D_marker_token_id is not None, "[unused1] 未在词汇表中定义"

    def tokenize(self, batch_text, add_special_tokens=False):
        """将一批文本转换为 token 列表。"""
        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]
        if add_special_tokens:
            prefix = [self.cls_token, self.D_marker_token]
            suffix = [self.sep_token]
            tokens = [prefix + lst + suffix for lst in tokens]
        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        """将一批文本转换为 token ID 列表。"""
        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']
        if add_special_tokens:
            prefix = [self.cls_token_id, self.D_marker_token_id]
            suffix = [self.sep_token_id]
            ids = [prefix + lst + suffix for lst in ids]
        return ids

    def tensorize(self, batch_text, bsize=None):
        """
        将一批文本进行分词、编码，并转换为 PyTorch 张量。
        这是最常用的接口。

        Args:
            batch_text (list[str]): 待处理的文档文本列表。
            bsize (int, optional): 如果提供，则会将结果按长度排序并划分为多个批次。

        Returns:
            如果 bsize 未提供，返回 (ids, mask) 元组。
            如果 bsize 提供，返回 (batches, reverse_indices) 元组。
        """
        # 在文本前添加一个占位符，以便后续插入 [D] 标记
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(
            batch_text,
            padding='longest',
            truncation='longest_first',
            return_tensors='pt',
            max_length=self.doc_maxlen
        )
        ids, mask = obj['input_ids'], obj['attention_mask']

        # 将占位符对应的 token ID 替换为 [D] 标记的 ID
        ids[:, 1] = self.D_marker_token_id

        if bsize:
            # 按长度排序并分批，这可以提高训练/推理效率
            ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
            batches = _split_into_batches(ids, mask, bsize)
            return batches, reverse_indices

        return ids, mask