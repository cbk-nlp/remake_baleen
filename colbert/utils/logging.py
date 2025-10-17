# 文件名: colbert/utils/logging.py

import os
import sys
import ujson
import traceback

from colbert.utils.utils import print_message, create_directory


class Logger:
    """
    一个用于实验日志记录的类。

    它负责将训练过程中的各种信息（如命令行参数、异常信息、警告）
    保存到实验目录下的 `logs/` 子目录中。
    在分布式环境中，通常只有主进程（rank 0）会执行实际的文件写入操作。

    注意: 代码中包含了对 MLflow 和 TensorBoard 的注释掉的集成代码，
          表明该类曾设计为与这些实验跟踪工具配合使用。
    """

    def __init__(self, rank, run):
        """
        初始化 Logger。

        Args:
            rank (int): 当前进程的排名。
            run (Run): 全局的实验运行对象。
        """
        self.rank = rank
        self.is_main = self.rank in [-1, 0] # 判断是否为主进程
        self.run = run
        self.logs_path = os.path.join(self.run.path, "logs/")

        if self.is_main:
            # 只有主进程创建日志目录
            create_directory(self.logs_path)

    def _log_exception(self, etype, value, tb):
        """记录捕获到的异常信息到 exception.txt。"""
        if not self.is_main:
            return

        output_path = os.path.join(self.logs_path, 'exception.txt')
        trace = ''.join(traceback.format_exception(etype, value, tb)) + '\n'
        print_message(trace, '\n\n')
        self.log_new_artifact(output_path, trace)

    def _log_args(self, args):
        """记录启动实验时使用的命令行参数。"""
        if not self.is_main:
            return
            
        # 将原始的命令行参数保存到 args.txt
        with open(os.path.join(self.logs_path, 'args.txt'), 'w') as f:
            f.write(' '.join(sys.argv) + '\n')
            
        # 将解析后的参数对象以 JSON 格式保存到 args.json
        with open(os.path.join(self.logs_path, 'args.json'), 'w') as f:
            # 注意: args.input_arguments 是一个深拷贝，只包含用户输入的参数
            ujson.dump(args.input_arguments.__dict__, f, indent=4)

    def log_new_artifact(self, path, content):
        """将指定内容写入一个新的日志文件。"""
        with open(path, 'w') as f:
            f.write(content)

    def warn(self, *args):
        """记录警告信息到 warnings.txt。"""
        msg = print_message('[警告]', '\t', *args)
        if self.is_main:
            with open(os.path.join(self.logs_path, 'warnings.txt'), 'a') as f:
                f.write(msg + '\n\n\n')

    def info_all(self, *args):
        """在所有进程中都打印信息，并带上 rank 前缀。"""
        print_message(f'[{self.rank}]', '\t', *args)

    def info(self, *args):
        """只在主进程中打印信息。"""
        if self.is_main:
            print_message(*args)