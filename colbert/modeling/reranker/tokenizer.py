# 文件名: colbert/modeling/reranker/tokenizer.py

from transformers import AutoTokenizer

class RerankerTokenizer:
    """
    为 ElectraReranker 设计的专用分词器。

    它的主要任务是将一个查询和一个文档拼接成一个单一的输入序列，
    格式通常是 `[CLS] query [SEP] passage [SEP]`。
    """
    def __init__(self, total_maxlen, base):
        """
        初始化 RerankerTokenizer。

        Args:
            total_maxlen (int): 拼接后序列的最大长度。
            base (str): 基础模型的名称 (例如 'google/electra-base-discriminator')。
        """
        self.total_maxlen = total_maxlen
        self.tok = AutoTokenizer.from_pretrained(base)

    def tensorize(self, questions, passages):
        """
        将一批查询和段落进行分词、拼接和张量化。

        Args:
            questions (list[str]): 查询字符串列表。
            passages (list[str]): 段落字符串列表。

        Returns:
            dict: HuggingFace Tokenizer 的输出，包含 'input_ids', 'attention_mask' 等。
        """
        assert isinstance(questions, (list, tuple))
        assert isinstance(passages, (list, tuple))

        # HuggingFace Tokenizer 可以直接处理成对的输入
        encoding = self.tok(
            questions,
            passages,
            padding='longest',          # 填充到批次中最长的序列长度
            truncation='longest_first', # 如果超长，优先截断较长的序列（通常是 passage）
            return_tensors='pt',        # 返回 PyTorch 张量
            max_length=self.total_maxlen
        )

        return encoding