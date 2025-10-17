# 文件名: colbert/data/__init__.py
# 作用:
# 这是 data 子包的初始化文件。
# 它导入了该目录下的所有主要数据处理类（Collection, Queries, Ranking, Examples），
# 使得这些类可以作为 colbert.data 模块的一部分被外部调用，为数据加载和管理提供了一个统一的接口。

from .collection import *
from .queries import *

from .ranking import *
from .examples import *