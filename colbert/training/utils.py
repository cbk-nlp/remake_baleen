# 文件名: colbert/training/utils.py

import os
import torch

# 导入 ColBERT 的基础设施和参数模块
from colbert.infra.run import Run
from colbert.parameters import SAVED_CHECKPOINTS


def print_progress(scores):
    """
    打印当前批次中正负样本的平均得分，以及它们之间的差距。
    这对于监控训练过程非常有用。

    Args:
        scores (torch.Tensor): 形状为 (bsize, nway) 的分数张量，
                               其中第一列是正样本的分数。
    """
    # 计算正样本和负样本的平均分
    positive_avg = scores[:, 0].mean().item()
    negative_avg = scores[:, 1:].mean().item()
    
    # 格式化输出
    print(f"#>>>   正例平均分: {positive_avg:.2f}, 负例平均分: {negative_avg:.2f}, \t"
          f"差距: {positive_avg - negative_avg:.2f}")


def manage_checkpoints(config, model, optimizer, batch_idx, consumed_all_triples=False):
    """
    管理模型的保存（创建检查点）。

    该函数会根据当前的迭代步数（batch_idx）和预设的保存点列表 `SAVED_CHECKPOINTS`
    来决定是否要保存模型。它也会在训练结束时强制保存一个最终的模型。

    Args:
        config (ColBERTConfig): 训练配置。
        model (torch.nn.Module): 待保存的模型。
        optimizer (torch.optim.Optimizer): 优化器状态也需要被保存。
        batch_idx (int): 当前的训练步数。
        consumed_all_triples (bool, optional): 是否已处理完所有训练数据。默认为 False。
    
    Returns:
        str or None: 如果进行了保存，则返回检查点的路径；否则返回 None。
    """
    # 确定检查点的保存目录
    checkpoints_path = os.path.join(Run().path_, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # 检查是否需要保存
    path_to_save = None
    # 1. 每隔一定步数（例如 2000 步）保存一次
    if batch_idx % 2000 == 0:
        path_to_save = os.path.join(checkpoints_path, "colbert")
    # 2. 如果当前步数在预设的 `SAVED_CHECKPOINTS` 列表中
    if batch_idx in SAVED_CHECKPOINTS:
        path_to_save = os.path.join(checkpoints_path, f"colbert-{batch_idx}")
    # 3. 如果所有数据都已处理完毕
    if consumed_all_triples:
        path_to_save = os.path.join(checkpoints_path, "colbert.final")

    if path_to_save:
        # 获取模型（处理 DistributedDataParallel 包装）
        model_to_save = model.module if hasattr(model, 'module') else model
        
        print(f"#> 正在保存检查点至 {path_to_save} ...")
        
        # 调用模型自身的 save 方法，该方法会保存模型、分词器和配置
        model_to_save.save(path_to_save)
        
        # 额外保存优化器状态和当前步数
        optimizer_path = os.path.join(path_to_save, 'optimizer.pt')
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'batch': batch_idx,
        }, optimizer_path)

        return path_to_save

    return None