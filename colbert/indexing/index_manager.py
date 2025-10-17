# 文件名: colbert/indexing/index_manager.py

import torch
import numpy as np
from bitarray import bitarray


class IndexManager:
    """
    一个简单的索引管理类，提供保存张量和 bitarray 的基本功能。
    
    注意: bitarray 相关的功能在当前版本的代码中可能已不常用，
    主要依赖 torch.save。
    """

    def __init__(self, dim):
        """
        初始化 IndexManager。

        Args:
            dim (int): 嵌入向量的维度。
        """
        self.dim = dim

    def save(self, tensor, path_prefix):
        """使用 torch.save 保存一个张量。"""
        torch.save(tensor, path_prefix)

    def save_bitarray(self, bitarray_obj, path_prefix):
        """将一个 bitarray 对象写入文件。"""
        with open(path_prefix, "wb") as f:
            bitarray_obj.tofile(f)


def load_index_part(filename, verbose=True):
    """
    从文件加载索引的一部分。
    为了向后兼容，它能处理保存为列表格式的旧索引。

    Args:
        filename (str): 索引部分文件的路径。
        verbose (bool, optional): 是否打印加载信息。默认为 True。

    Returns:
        torch.Tensor: 加载的张量。
    """
    part = torch.load(filename)

    # 处理旧格式的索引（保存为 list of tensors）
    if isinstance(part, list):
        part = torch.cat(part)

    return part


def load_compressed_index_part(filename, dim, bits):
    """
    加载由 bitarray 保存的压缩索引部分。
    这个函数主要用于向后兼容或特定的压缩格式。

    Args:
        filename (str): 文件路径。
        dim (int): 向量维度。
        bits (int): 每个维度值的比特数。

    Returns:
        torch.Tensor: 解码后的索引部分张量。
    """
    a = bitarray()
    with open(filename, "rb") as f:
        a.fromfile(f)

    n = len(a) // (dim * bits)
    # 将二进制数据重新解释为 uint8 类型的 numpy 数组，然后转换为 torch 张量
    part = torch.tensor(np.frombuffer(a.tobytes(), dtype=np.uint8))
    part = part.reshape((n, int(np.ceil(dim * bits / 8))))

    return part