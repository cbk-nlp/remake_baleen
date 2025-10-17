# 文件名: colbert/infra/run.py

import os
import atexit
from contextlib import contextmanager

from colbert.utils.utils import create_directory, print_message
from colbert.infra.config import RunConfig


class Run:
    """
    一个全局单例（Singleton）对象，用于管理整个实验的运行上下文。

    它维护一个配置栈，允许通过 `with Run().context(...)` 语句临时
    更改配置。它还提供了全局访问 rank、nranks、实验路径等信息
    的方法，并封装了日志记录和文件操作。
    """
    _instance = None
    # 默认启用 tokenizers 库的并行化以提高性能
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def __new__(cls):
        """
        实现单例模式。
        在第一次创建时，初始化一个默认的 RunConfig。
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stack = []
            
            # 创建并设置一个默认的全局运行配置
            run_config = RunConfig()
            run_config.assign_defaults()
            cls._instance.__append(run_config)
        return cls._instance

    @property
    def config(self):
        """返回当前上下文中最顶层的配置对象。"""
        return self.stack[-1]

    def __getattr__(self, name):
        """
        使得可以直接通过 `Run().xxx` 的方式访问当前配置对象的属性。
        例如 `Run().rank` 相当于 `Run().config.rank`。
        """
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __append(self, runconfig: RunConfig):
        """将一个新的配置对象压入栈顶。"""
        self.stack.append(runconfig)

    def __pop(self):
        """从栈顶弹出一个配置对象。"""
        self.stack.pop()

    @contextmanager
    def context(self, runconfig: RunConfig, inherit_config=True):
        """
        一个上下文管理器，用于临时应用一个新的配置。
        在 `with` 语句块内，新的配置生效；退出时，恢复到之前的配置。

        Args:
            runconfig (RunConfig): 要应用的新配置。
            inherit_config (bool, optional): 如果为 True，新配置将继承当前配置的设置，
                                             并用自己的设置进行覆盖。默认为 True。
        """
        if inherit_config:
            runconfig = RunConfig.from_existing(self.config, runconfig)
        self.__append(runconfig)
        try:
            yield
        finally:
            self.__pop()
        
    def open(self, path, mode='r'):
        """
        打开一个位于当前实验路径下的文件。
        它会自动处理目录创建和覆盖检查。
        """
        # 将相对路径转换为实验目录下的绝对路径
        full_path = os.path.join(self.path_, path)
        
        # 如果目录不存在，则创建
        if ('w' in mode or 'a' in mode) and not os.path.exists(os.path.dirname(full_path)):
             create_directory(os.path.dirname(full_path))

        # 如果是写模式且不允许覆盖，则检查文件是否已存在
        if ('w' in mode) and not self.overwrite:
            assert not os.path.exists(full_path), f"文件 {full_path} 已存在，且不允许覆盖。"

        return open(full_path, mode=mode)
    
    def print(self, *args):
        """在分布式环境中，只在指定 rank 的进程中打印信息。"""
        print_message(f"[{self.rank}]", "\t\t", *args)

    def print_main(self, *args):
        """只在主进程 (rank 0) 中打印信息。"""
        if self.rank == 0:
            self.print(*args)