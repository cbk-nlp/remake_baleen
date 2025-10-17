# 文件名: colbert/modeling/__init__.py
# 作用:
# 这是 modeling 子包的初始化文件。
# 它导入了该目录下的所有主要与模型分词相关的类，
# 使得这些类可以作为 colbert.modeling.tokenization 模块的一部分被外部调用。
from colbert.modeling.tokenization.query_tokenization import *
from colbert.modeling.tokenization.doc_tokenization import *
from colbert.modeling.tokenization.utils import tensorize_triples