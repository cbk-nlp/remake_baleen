# 文件名: colbert/training/training.py

import time
import torch
import random
import torch.nn as nn
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.utils.amp import MixedPrecisionManager
from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker
from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.rerank_batcher import RerankBatcher
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


def train(config: ColBERTConfig, triples, queries=None, collection=None):
    """
    ColBERT 模型的核心训练函数。

    该函数由 `Launcher` 在每个分布式进程中调用。它负责：
    1.  设置随机种子和分布式环境。
    2.  初始化数据加载器 (Batcher)。
    3.  初始化模型、优化器和学习率调度器。
    4.  执行主训练循环，包括前向传播、损失计算、反向传播和参数更新。
    5.  管理检查点的保存。
    """
    # 确保有默认的检查点
    config.checkpoint = config.checkpoint or 'bert-base-uncased'

    # 只有主进程打印配置信息
    if config.rank < 1:
        config.help()

    # 设置随机种子以保证实验的可复现性
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    # 在分布式训练中，将总批次大小分配到每个进程
    assert config.bsize % config.nranks == 0, f"批次大小 {config.bsize} 必须能被进程数 {config.nranks} 整除"
    config.bsize = config.bsize // config.nranks
    print_message(f"每个进程使用的批次大小 = {config.bsize}, 梯度累积步数 = {config.accumsteps}")

    # 根据配置选择合适的数据加载器
    if config.reranker:
        reader = RerankBatcher(config, triples, queries, collection, config.rank, config.nranks)
    else:
        reader = LazyBatcher(config, triples, queries, collection, config.rank, config.nranks)

    # 根据配置初始化 ColBERT 或 Reranker 模型
    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    # 将模型移动到 GPU 并设置为训练模式
    colbert = colbert.to(DEVICE)
    colbert.train()

    # 使用 PyTorch 的 DistributedDataParallel 包装模型以支持分布式训练
    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)

    # 初始化优化器 (AdamW)
    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    # (可选) 初始化学习率调度器，实现 warmup 和线性衰减
    scheduler = None
    if config.warmup is not None:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)
    
    # 初始化自动混合精度（AMP）管理器
    amp = MixedPrecisionManager(config.amp)
    # 预先定义好标签，因为对于对比学习，正样本总是在第一个位置
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    train_loss = 0.0 # 用于平滑损失值
    start_batch_idx = 0 # 训练起始步数

    # --- 主训练循环 ---
    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        this_batch_loss = 0.0

        for batch in BatchSteps:
            with amp.context(): # 开启 AMP 上下文
                # --- 前向传播 ---
                if config.reranker:
                    encoding, target_scores = batch
                    scores = colbert(encoding.to(DEVICE))
                else: # ColBERT
                    queries, passages, target_scores = batch
                    scores = colbert(queries, passages)

                # --- 损失计算 ---
                scores = scores.view(-1, config.nway)
                
                # 如果有目标分数（用于知识蒸馏），则使用 KL 散度损失
                if len(target_scores) > 0 and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores, device=DEVICE).view(-1, config.nway)
                    target_scores = torch.nn.functional.log_softmax(target_scores * config.distillation_alpha, dim=-1)
                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else: # 否则，使用标准的交叉熵损失
                    loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                # 损失需要除以梯度累积步数
                loss = loss / config.accumsteps

            # 打印当前批次的分数信息
            if config.rank < 1:
                print_progress(scores)

            # --- 反向传播 ---
            amp.backward(loss)
            this_batch_loss += loss.item()

        # 平滑损失值
        train_loss = 0.999 * train_loss + 0.001 * this_batch_loss if batch_idx > 0 else this_batch_loss
        
        # --- 参数更新 ---
        amp.step(colbert, optimizer, scheduler)

        # --- 日志和检查点 ---
        if config.rank < 1:
            print_message(f"批次 {batch_idx+1} / {config.maxsteps} | 损失: {train_loss:.5f}")
            manage_checkpoints(config, colbert, optimizer, batch_idx + 1)

    # --- 训练结束 ---
    if config.rank < 1:
        print_message("#> 所有三元组处理完毕！")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx + 1, consumed_all_triples=True)
        return ckpt_path