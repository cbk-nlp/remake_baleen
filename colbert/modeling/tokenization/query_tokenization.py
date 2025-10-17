# 文件名: colbert/modeling/tokenization/query_tokenization.py

import torch
from colbert.modeling.hf_colbert import HF_ColBERT
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization.utils import _split_into_batches


class QueryTokenizer:
    """为 ColBERT 的查询（queries）设计的专用分词器。"""
    def __init__(self, config: ColBERTConfig):
        self.tok = HF_ColBERT.raw_tokenizer_from_pretrained(config.checkpoint)
        self.config = config
        self.query_maxlen = config.query_maxlen

        # 定义查询的特殊标记 [Q] 和用于填充的 [MASK]
        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        
        assert self.Q_marker_token_id is not None, "[unused0] 未在词汇表中定义"
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        """将一批查询文本转换为 token 列表。"""
        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]
        if add_special_tokens:
            prefix = [self.cls_token, self.Q_marker_token]
            suffix = [self.sep_token]
            # 用 [MASK] 标记填充到最大长度
            tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst) + 3)) for lst in tokens]
        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        """将一批查询文本转换为 token ID 列表。"""
        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']
        if add_special_tokens:
            prefix = [self.cls_token_id, self.Q_marker_token_id]
            suffix = [self.sep_token_id]
            ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3)) for lst in ids]
        return ids

    def tensorize(self, batch_text, bsize=None, context=None):
        """
        将一批查询文本（以及可选的上下文）进行分词、编码，并转换为 PyTorch 张量。
        
        一个关键特性是“查询增强”（query augmentation）：将查询填充到固定长度，
        并用 [MASK] 标记填充多余的部分。这在训练中被证明是有效的。
        """
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(
            batch_text,
            padding='max_length', # 填充到 max_length
            truncation=True,
            return_tensors='pt',
            max_length=self.query_maxlen
        )
        ids, mask = obj['input_ids'], obj['attention_mask']

        # 替换 [Q] 标记
        ids[:, 1] = self.Q_marker_token_id
        # 将所有的填充 token (ID=0) 替换为 [MASK] token
        ids[ids == 0] = self.mask_token_id

        # (可选) 如果提供了上下文，将其拼接到查询后面
        if context is not None:
            # ... (此处省略上下文拼接逻辑) ...
            pass

        # 根据配置决定是否在 attention mask 中关注 [MASK] token
        if self.config.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1

        if bsize:
            return _split_into_batches(ids, mask, bsize)
        
        # 第一次调用时打印示例，用于调试
        if not self.used:
            self.used = True
            print("\n#> QueryTokenizer.tensorize 的首次调用示例:")
            print(f"#> 输入文本: {batch_text[0]}")
            print(f"#> 输出 IDs: {ids[0]}")
            print(f"#> 输出 Mask: {mask[0]}\n")

        return ids, mask