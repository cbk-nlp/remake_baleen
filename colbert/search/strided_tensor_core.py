# 文件名: colbert/search/strided_tensor_core.py

import torch
import numpy as np

from colbert.utils.utils import flatten


class StridedTensorCore:
    """
    StridedTensor 的核心实现。

    它将一个由多个变长序列拼接而成的“扁平”张量（packed_tensor）和一个记录
    每个序列长度的 `lengths` 张量，转换为一种特殊的内存视图。
    这种视图允许我们像操作一个规则的二维数组一样，高效地索引和切片出
    任意一个原始的变长序列，而无需进行耗时的循环或复制。

    这是通过 PyTorch 底层的 `as_strided` 函数实现的，它改变了张量在内存中的
    “步长”（stride）信息，从而创建出数据重叠的视图。
    """

    def __init__(self, packed_tensor, lengths, dim=None):
        self.dim = dim
        self.tensor = packed_tensor
        self.inner_dims = self.tensor.size()[1:] # 记录除第一维外的其他维度

        self.lengths = lengths.long() if torch.is_tensor(lengths) else torch.LongTensor(lengths)

        # 为了优化，创建几个不同步长（stride）的视图
        # 查找时，会根据所需的最大长度选择一个最合适的（最小的）视图
        self.strides = _select_strides(self.lengths, [0.5, 0.75, 0.9, 0.95]) + [self.lengths.max().item()]
        self.max_stride = self.strides[-1]

        # 计算每个序列在扁平张量中的起始偏移量
        zero = torch.zeros(1, dtype=torch.long, device=self.lengths.device)
        self.offsets = torch.cat((zero, torch.cumsum(self.lengths, dim=0)))

        # 添加一些填充，以防止在创建视图时发生越界访问
        if self.offsets[-2] + self.max_stride > self.tensor.size(0):
            padding_size = self.max_stride
            padding = torch.zeros(padding_size, *self.inner_dims, dtype=self.tensor.dtype, device=self.tensor.device)
            self.tensor = torch.cat((self.tensor, padding))

        # 为每个步长预先创建内存视图
        self.views = {stride: _create_view(self.tensor, stride, self.inner_dims) for stride in self.strides}

    def as_padded_tensor(self):
        """
        将整个 StridedTensor 转换为一个大的、填充过的常规张量和一个掩码。
        这在需要将数据送入期望固定尺寸输入的模型时很有用。
        """
        view = self.views[self.max_stride][self.offsets[:-1]]
        mask = _create_mask(self.lengths, self.max_stride, like=view)
        return view, mask

# --- 辅助函数 ---

def _select_strides(lengths, quantiles):
    """根据长度分布的分位数来选择一组有代表性的步长。"""
    if lengths.size(0) < 5_000:
        # 如果序列总数较少，直接在所有长度上计算分位数
        return torch.quantile(lengths.float(), torch.tensor(quantiles, device=lengths.device)).int().tolist()
    # 否则，随机采样一小部分长度来估算分位数，以提高效率
    sample_indices = torch.randint(0, lengths.size(0), size=(2_000,))
    return torch.quantile(lengths[sample_indices].float(), torch.tensor(quantiles, device=lengths.device)).int().tolist()


def _create_view(tensor, stride, inner_dims):
    """
    使用 `torch.as_strided` 创建核心的内存视图。
    
    Args:
        tensor (torch.Tensor): 扁平化的一维或多维张量。
        stride (int): 视图的第二维（即窗口）的大小。
        inner_dims (tuple): 原始张量除第一维外的其他维度。
    
    Returns:
        torch.Tensor: 一个新的张量视图。
    """

    # 计算输出视图的大小和步长
    outdim = tensor.size(0) - stride + 1
    size = (outdim, stride, *inner_dims)
    
    inner_dim_prod = int(np.prod(inner_dims)) if inner_dims else 1
    # 关键步骤：定义新的步长
    # 第一个维度移动一个元素，指针移动 `inner_dim_prod` 个字节
    # 第二个维度移动一个元素，指针也移动 `inner_dim_prod` 个字节
    # 这就造成了数据在第二个维度上的重叠
    multidim_stride = [inner_dim_prod, inner_dim_prod] + list(tensor.stride()[1:])

    return torch.as_strided(tensor, size=size, stride=multidim_stride)


def _create_mask(lengths, stride, like=None):
    """创建一个布尔掩码，用于标识填充部分。"""
    mask = (torch.arange(stride, device=lengths.device) < lengths.unsqueeze(-1))
    if like is not None:
        # 扩展掩码的维度以匹配目标张量
        for _ in range(like.dim() - mask.dim()):
            mask = mask.unsqueeze(-1)
    return mask