# 文件名: colbert/search/strided_tensor.py

import torch
from .strided_tensor_core import StridedTensorCore, _create_mask


class StridedTensor(StridedTensorCore):
    """
    StridedTensor 的高级接口。

    它继承自 StridedTensorCore，并提供了更方便的 `lookup` 方法，
    允许用户通过索引（例如，一批文档 ID）来高效地提取对应的
    多个变长序列。
    """
    def __init__(self, packed_tensor, lengths, dim=None):
        super().__init__(packed_tensor, lengths, dim=dim)

    def _prepare_lookup(self, pids):
        """准备查找操作所需的 pids, lengths, 和 offsets。"""
        if isinstance(pids, list):
            pids = torch.tensor(pids)
        assert pids.dim() == 1

        pids = pids.cuda().long()
        lengths = self.lengths[pids].cuda()
        offsets = self.offsets[pids]
        return pids, lengths, offsets

    def lookup(self, pids, output='packed'):
        """
        根据给定的索引 (pids) 查找对应的序列。

        Args:
            pids (list or torch.Tensor): 待查找的序列索引。
            output (str, optional): 输出格式。
                                    'packed': 返回一个扁平化的张量和每个序列的长度。
                                    'padded': 返回一个填充过的张量和一个掩码。
                                    默认为 'packed'。

        Returns:
            根据 `output` 的设置返回不同格式的结果。
        """
        pids, lengths, offsets = self._prepare_lookup(pids)

        # 找到能容纳所有待查找序列的最小步长
        max_len = lengths.max().item() if len(lengths) > 0 else 0
        stride = next((s for s in self.strides if max_len <= s), self.max_stride)

        # 从预先创建的视图中直接切片，这是最高效的部分
        tensor_view = self.views[stride][offsets].cuda()

        # 创建掩码
        mask = _create_mask(lengths, stride)

        if output == 'padded':
            return tensor_view, mask
        
        assert output == 'packed'
        # 应用掩码，只保留有效部分，返回扁平化的张量
        tensor_packed = tensor_view[mask]
        return tensor_packed, lengths