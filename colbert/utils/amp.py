# 文件名: colbert/utils/amp.py

import torch
from contextlib import contextmanager


# 定义一个空的上下文管理器，用于在不激活 AMP 时提供一个兼容的接口
class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


class MixedPrecisionManager:
    """
    一个用于管理 PyTorch 自动混合精度（AMP）训练的辅助类。

    AMP 允许在训练过程中自动使用半精度（FP16）和单精度（FP32）浮点数，
    从而在不显著降低模型精度的情况下，减少显存占用并加速训练。
    这个类封装了 AMP 的核心组件：`GradScaler` 和 `autocast` 上下文。
    """

    def __init__(self, activated):
        """
        初始化 MixedPrecisionManager。

        Args:
            activated (bool): 是否激活混合精度训练。
        """
        self.activated = activated

        if self.activated:
            # GradScaler 用于在反向传播时自动缩放损失，以防止半精度下的梯度下溢问题
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        """
        返回一个上下文管理器。
        如果 AMP 被激活，则返回 `torch.cuda.amp.autocast()`，它会自动将
        上下文内的 CUDA 操作转换为半精度。
        如果未激活，则返回一个空的上下文管理器。
        """
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        """
        执行反向传播。

        如果 AMP 被激活，它会使用 `GradScaler` 来缩放损失，然后再进行反向传播。
        否则，它只执行常规的 `loss.backward()`。
        """
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, model, optimizer, scheduler=None):
        """
        执行优化器步骤（参数更新）。

        如果 AMP 被激活，它会包含额外的步骤：
        1.  `unscale_`: 在梯度裁剪前，将梯度从缩放后的值还原回来。
        2.  `clip_grad_norm_`: 对梯度进行裁剪，防止梯度爆炸。
        3.  `step`: 执行参数更新。如果遇到非法的梯度（NaN 或 inf），会自动跳过。
        4.  `update`: 更新 `GradScaler` 的缩放因子。

        Args:
            model (torch.nn.Module): 待训练的模型。
            optimizer (torch.optim.Optimizer): 优化器。
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器。
        """
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()