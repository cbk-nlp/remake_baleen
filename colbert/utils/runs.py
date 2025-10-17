# 文件名: colbert/utils/runs.py

import os
import sys
import time
import __main__
import traceback
from contextlib import contextmanager

import colbert.utils.distributed as distributed
from colbert.utils.logging import Logger
from colbert.utils.utils import timestamp, create_directory, print_message

# 注意: 这个文件是 infra/run.py 的一个早期版本。
# 现代 ColBERT 代码库主要使用 infra/run.py 中的 Run 单例。
# 这个文件可能为了向后兼容而保留。


class _RunManager:
    """
    一个用于管理实验运行生命周期的类。

    它负责初始化实验目录结构、设置日志记录器、捕获异常，并在实验
    结束时记录最终状态。
    """

    def __init__(self):
        self.path = None
        self.name = timestamp() # 默认运行名称为当前时间戳
        self.exit_status = '完成' # 默认为成功完成

        self._logger = None
        self.start_time = time.time()

    def init(self, rank, root, experiment, name):
        """初始化实验目录和日志记录器。"""
        self.experiments_root = os.path.abspath(root)
        self.experiment = experiment
        self.name = name
        script_name = os.path.basename(__main__.__file__) if '__file__' in dir(__main__) else 'none'
        self.path = os.path.join(self.experiments_root, self.experiment, script_name, self.name)

        # 主进程负责创建目录和处理覆盖逻辑
        if rank < 1:
            if os.path.exists(self.path):
                response = input(f"路径 {self.path} 已存在。是否覆盖? (yes/no): ")
                if response.strip().lower() != 'yes':
                    sys.exit("用户选择不覆盖，程序退出。")
            else:
                create_directory(self.path)

        distributed.barrier(rank) # 等待所有进程同步

        # 初始化日志记录器
        self._logger = Logger(rank, self)
        # 将 logger 的方法暴露出来以便直接调用
        self._log_args = self._logger._log_args
        self.warn = self._logger.warn
        self.info = self._logger.info
        self.info_all = self._logger.info_all
        
    @contextmanager
    def context(self, consider_failed_if_interrupted=True):
        """
        一个上下文管理器，用于包装实验的主要逻辑。
        它会自动捕获异常并记录实验的最终状态。
        """
        try:
            yield
        except KeyboardInterrupt:
            print('\n\n用户中断\n\n')
            self.exit_status = '被终止'
            sys.exit(130) # Ctrl+C 的标准退出码
        except Exception as ex:
            # 记录异常信息
            self._logger._log_exception(type(ex), ex, ex.__traceback__)
            self.exit_status = '失败'
            raise
        finally:
            # 记录总耗时
            total_seconds = str(time.time() - self.start_time)
            self._logger.log_new_artifact(os.path.join(self._logger.logs_path, 'elapsed.txt'), total_seconds)
            print(f"实验 {self.name} 状态: {self.exit_status}，总耗时: {total_seconds:.2f} 秒。")

# 创建一个全局的 RunManager 实例，以便在代码库的任何地方访问
Run = _RunManager()