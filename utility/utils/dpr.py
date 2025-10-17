# 文件名: utility/utils/dpr.py

"""
源代码来自 Facebook Research 的 DPR 实现：
https://github.com/facebookresearch/DPR/tree/master/dpr
"""

import unicodedata
import regex as re # 使用更强大的 regex 库代替 re


class SimpleTokenizer:
    """
    一个简单的分词器，用于将文本分割为单词（token）。
    它主要通过正则表达式来匹配字母数字序列和其他非空白字符。
    """
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        self._regexp = re.compile(f'({self.ALPHA_NUM})|({self.NON_WS})',
                                  flags=re.IGNORECASE | re.UNICODE | re.MULTILINE)

    def tokenize(self, text):
        # 查找所有匹配项并返回它们的文本内容
        return [m.group() for m in self._regexp.finditer(text)]

# 创建一个全局的分词器实例
_tokenizer = SimpleTokenizer()


def DPR_normalize(text):
    """
    对文本进行 DPR 风格的标准化。
    这通常包括：
    1.  Unicode NFD 标准化（将字符和其重音符号分离）。
    2.  转换为小写。
    3.  使用 SimpleTokenizer 进行分词。

    Args:
        text (str): 输入文本。

    Returns:
        list[str]: 标准化后的 token 列表。
    """
    # unicodedata.normalize('NFD', text) 是处理多语言文本（特别是带重音符号的）的关键步骤
    return _tokenizer.tokenize(unicodedata.normalize('NFD', text).lower())


def has_answer(tokenized_answers, text):
    """
    检查给定的文本中是否精确包含了任何一个标准答案。

    Args:
        tokenized_answers (list[list[str]]): 一个列表，其中每个元素都是一个
                                             经过 DPR_normalize 处理的答案 token 列表。
        text (str): 待检查的段落文本。

    Returns:
        bool: 如果找到了精确匹配，则返回 True，否则返回 False。
    """
    # 对段落文本也进行同样的标准化处理
    normalized_text = DPR_normalize(text)
    
    # 遍历所有可能的答案
    for single_answer_tokens in tokenized_answers:
        # 使用滑动窗口在标准化后的文本中查找答案 token 序列
        for i in range(len(normalized_text) - len(single_answer_tokens) + 1):
            if single_answer_tokens == normalized_text[i : i + len(single_answer_tokens)]:
                return True # 找到匹配，立即返回
    
    return False