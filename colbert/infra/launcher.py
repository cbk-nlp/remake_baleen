# 文件名: colbert/infra/launcher.py

import os
import time
import torch
import random
import numpy as np
import torch.multiprocessing as mp

# 导入分布式工具函数
import colbert.utils.distributed as distributed
# 导入实验运行和配置管理类
from colbert.infra.run import Run
from colbert.infra.config import BaseConfig, RunConfig, RunSettings
# 导入工具函数
from colbert.utils.utils import print_message


# 尝试设置多进程的启动方法为 'spawn'，这在 CUDA 环境下通常更稳定
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


class Launcher:
    """
    一个用于启动分布式（多进程）任务的启动器。

    它负责创建和管理多个子进程，并在每个子进程中运行一个给定的函数（callee）。
    这对于在多个 GPU 上并行执行训练或索引等任务至关重要。
    """
    def __init__(self, callee, run_config=None, return_all=False):
        """
        初始化 Launcher。

        Args:
            callee (function): 要在每个子进程中执行的目标函数。
            run_config (RunConfig, optional): 运行配置。
            return_all (bool, optional): 如果为 True，则返回所有子进程的返回值列表；
                                        否则只返回第一个子进程的返回值。默认为 False。
        """
        self.callee = callee
        self.return_all = return_all
        self.run_config = RunConfig.from_existing(Run().config, run_config)
        self.nranks = self.run_config.nranks

    def launch(self, custom_config, *args):
        """
        启动分布式任务。

        Args:
            custom_config (BaseConfig): 特定于此次任务的配置。
            *args: 传递给目标函数 `callee` 的额外参数。

        Returns:
            根据 `return_all` 的设置，返回一个或所有子进程的返回值。
        """
        # 创建一个队列，用于从子进程接收返回值
        return_value_queue = mp.Queue()
        # 随机化端口号，以避免同时启动多个任务时发生冲突
        port = str(12355 + random.randint(0, 1000))

        all_procs = []
        for new_rank in range(self.nranks):
            # 为每个子进程创建一个独立的、包含了 rank 信息的配置对象
            new_config = type(custom_config).from_existing(custom_config, self.run_config, RunConfig(rank=new_rank))
            # 准备传递给子进程设置函数的参数
            proc_args = (self.callee, port, return_value_queue, new_config, *args)
            all_procs.append(mp.Process(target=setup_new_process, args=proc_args))

        # 清空 GPU 缓存，为新进程释放显存
        torch.cuda.empty_cache()

        # 启动所有子进程
        for proc in all_procs:
            proc.start()

        # 从队列中收集所有子进程的返回值
        return_values = sorted([return_value_queue.get() for _ in all_procs])
        return_values = [val for rank, val in return_values]

        if not self.return_all:
            return_values = return_values[0]
        
        # 等待所有子进程结束
        for proc in all_procs:
            proc.join()
            print("#> 子进程已结束...")
        
        return return_values


def setup_new_process(callee, port, return_value_queue, config, *args):
    """
    在新的子进程中执行的设置函数。

    它负责：
    1.  设置随机种子以保证可复现性。
    2.  设置分布式通信所需的环境变量。
    3.  初始化 PyTorch 分布式后端。
    4.  在 Run 上下文中执行目标函数 `callee`。
    5.  将 `callee` 的返回值放入队列中。
    """
    # 设置随机种子
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    rank, nranks = config.rank, config.nranks

    # 设置分布式通信所需的环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["WORLD_SIZE"] = str(config.nranks)
    os.environ["RANK"] = str(config.rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.gpus_[:nranks]))

    # 初始化分布式后端
    distributed.init(rank)

    # 在新的 Run 上下文中执行目标函数
    with Run().context(config, inherit_config=False):
        return_val = callee(config, *args)

    # 将返回值和 rank 一起放入队列
    return_value_queue.put((rank, return_val))