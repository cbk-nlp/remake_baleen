# 文件名: baleen/condenser/tokenization.py

import torch
from transformers import ElectraTokenizerFast


class AnswerAwareTokenizer:
    """
    一个为 Condenser 模型设计的、能够感知答案位置的分词器。

    它不仅负责将查询和段落拼接并编码，还负责识别段落中所有可能的
    候选答案片段（通常由 `[MASK]` 标记界定），并将它们的位置信息
    编码为模型可以处理的格式。
    """
    def __init__(self, total_maxlen, bert_model='google/electra-base-discriminator'):
        self.total_maxlen = total_maxlen
        self.tok = ElectraTokenizerFast.from_pretrained(bert_model)

    def process(self, questions, passages):
        """处理一批查询和段落，返回一个包含所有编码信息的对象。"""
        return TokenizationObject(self, questions, passages)

    def tensorize(self, questions, passages):
        """
        将查询和段落拼接、分词并转换为张量。
        格式: `[CLS] question [SEP] passage [SEP]`
        """
        encoding = self.tok(
            questions,
            passages,
            padding='longest',
            truncation='longest_first',
            return_tensors='pt',
            max_length=self.total_maxlen
        )
        return encoding


class TokenizationObject:
    """
    一个数据类，用于封装 `AnswerAwareTokenizer` 处理后的所有信息。
    """
    def __init__(self, tokenizer: AnswerAwareTokenizer, questions, passages):
        assert isinstance(questions, list) and isinstance(passages, list)
        assert len(questions) == 1 or len(questions) == len(passages)

        self.tok = tokenizer
        # 如果只有一个查询，则复制它以匹配所有段落
        self.questions = questions if len(questions) == len(passages) else questions * len(passages)
        self.passages = passages
        
        # 执行核心的编码操作
        self.encoding = self._encode()

    def _encode(self):
        """调用分词器进行张量化。"""
        return self.tok.tensorize(self.questions, self.passages)