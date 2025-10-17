# 文件名: colbert/infra/__init__.py
# 作用:
# 这是 infra 子包的初始化文件。
# 它从子模块中导入了核心的 Run, ColBERTConfig, RunConfig 等类，
# 使得这些基础设施类可以方便地通过 `from colbert.infra import ...` 来访问。

from .run import *
from .config import *