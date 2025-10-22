# 文件名: colbert/utils/distributed.py

import os
import torch

# 一个全局标志，用于防止重复初始化
ALREADY_INITIALIZED = False


def init(rank):
    """
    初始化 PyTorch 的分布式通信后端 (NCCL)。

    这个函数会根据环境变量（特别是 `WORLD_SIZE`）来判断是否处于分布式环境中。
    如果是，它会设置当前进程应该使用的 GPU，并调用 `torch.distributed.init_process_group`
    来建立进程间的通信。

    Args:
        rank (int): 当前进程的全局排名 (global rank)。

    Returns:
        tuple:
            - nranks (int): 总的进程数 (world size)。
            - is_distributed (bool): 是否成功初始化了分布式环境。
    """
    global ALREADY_INITIALIZED
    if ALREADY_INITIALIZED:
        nranks = int(os.environ.get('WORLD_SIZE', 1))
        return nranks, nranks > 1

    ALREADY_INITIALIZED = True

    # 从环境变量中获取总进程数
    nranks = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = nranks > 1

    if is_distributed:
        num_gpus = torch.cuda.device_count()
        print(f'总进程数 = {nranks} \t GPU数量 = {num_gpus} \t 当前进程 (rank {rank}) 使用 device={(rank + 1) % num_gpus}')
        
        # 为当前进程设置默认的 GPU 设备
        torch.cuda.set_device((rank + 1) % num_gpus)
        
        # 初始化进程组，使用 NCCL 后端进行 GPU 间的高效通信
        # 'env://' 初始化方法会从环境变量中自动读取 MASTER_ADDR 和 MASTER_PORT
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    return nranks, is_distributed


def barrier(rank):
    """
    在分布式环境中的所有进程之间设置一个同步点（屏障）。

    调用此函数后，所有进程都会暂停，直到其他所有进程也到达这个点，
    然后才会一起继续执行。这对于确保某些操作（例如文件写入、模型加载）
    在所有进程中都完成后再继续下一步非常重要。

    Args:
        rank (int): 当前进程的排名。
    """
    nranks = int(os.environ.get('WORLD_SIZE', 1))

    if rank >= 0 and nranks > 1:
        torch.distributed.barrier(device_ids=[(rank + 1) % torch.cuda.device_count()])