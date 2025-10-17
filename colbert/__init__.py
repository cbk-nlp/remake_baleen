# 文件名: colbert/__init__.py
# 作用:
# 这是一个顶层的包初始化文件。
# 它将 colbert 包中的主要类（如 Trainer, Indexer, Searcher, Checkpoint）导入到包的命名空间中，
# 使得用户可以直接从 colbert 导入这些核心功能，简化了API的使用。

from .trainer import Trainer
from .indexer import Indexer
from .searcher import Searcher

from .modeling.checkpoint import Checkpoint