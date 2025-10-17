# 文件名: utility/supervision/triples.py

import os
import random
import ujson
from argparse import ArgumentParser

from colbert.utils.utils import print_message, load_ranking, groupby_first_item, create_directory
from utility.utils.save_metadata import save_metadata

# 定义生成三元组数量的上限
MAX_NUM_TRIPLES = 40_000_000


def sample_negatives(negatives, num_sampled, biased=None):
    """
    从给定的负例列表中随机采样。

    Args:
        negatives (list): 负例 PID 列表。
        num_sampled (int): 要采样的数量。
        biased (int, optional): 如果提供（例如 100 或 200），则启用偏置采样，
                                即 50% 的负例从排名前 `biased` 的负例中采样，
                                另外 50% 从其余负例中采样。这有助于采样到更具挑战性的“难负例”。
    """
    num_sampled = min(len(negatives), num_sampled)

    if biased and num_sampled < len(negatives) and (num_sampled % 2 == 0):
        num_sampled_top = num_sampled // 2
        num_sampled_rest = num_sampled - num_sampled_top
        
        oversampled = negatives[:biased]
        undersampled = negatives[biased:]

        if len(oversampled) >= num_sampled_top and len(undersampled) >= num_sampled_rest:
            return random.sample(oversampled, num_sampled_top) + random.sample(undersampled, num_sampled_rest)

    return random.sample(negatives, num_sampled)


def sample_for_query(qid, ranking, args_positives, depth, permissive, biased):
    """
    为单个查询从其带标注的排序列表中采样生成三元组。

    Args:
        qid (int): 查询 ID。
        ranking (list): 该查询的带标注排序结果，格式为 `[(pid, rank, ..., label), ...]`。
        args_positives (list[tuple]): 定义如何选择正例的策略，例如 `[(5, 50)]` 表示
                                     “从排名前 50 的段落中，最多选择 5 个正例”。
        depth (int): 负例采样的最大深度。
        permissive (bool): 如果为 True，即使是排名较低的正例也会被考虑，但会与较少的负例配对。
        biased (int): 传递给 `sample_negatives` 的偏置采样参数。

    Returns:
        list[tuple]: 为该查询生成的三元组列表。
    """
    positives, negatives, triples = [], [], []

    # 1. 遍历排序列表，根据标签分离正例和负例
    for pid, rank, *_, label in ranking:
        assert rank >= 1, f"排名应从 1 开始，但得到了 {rank}"
        assert label in [0, 1]

        if rank > depth:
            break

        if label == 1:
            # 根据 `args_positives` 策略决定是否采纳这个正例
            is_high_rank_positive = any(rank <= maxDepth and len(positives) < maxBest for maxBest, maxDepth in args_positives)
            if is_high_rank_positive:
                positives.append((pid, 0)) # 0 表示这是一个高质量正例，应与更多负例配对
            elif permissive:
                positives.append((pid, rank)) # permissive 模式下，低质量正例也保留，但与较少负例配对
        else:
            negatives.append(pid)

    # 2. 为每个选中的正例采样负例，构建三元组
    for pos_pid, neg_start_rank in positives:
        # 高质量正例 (neg_start_rank=0) 采样更多负例
        num_sampled = 100 if neg_start_rank == 0 else 5
        
        # `neg_start_rank` 允许我们只从比当前正例排名更低的段落中采样负例
        negatives_to_sample_from = negatives[neg_start_rank:]
        
        biased_mode = biased if neg_start_rank == 0 else None
        for neg_pid in sample_negatives(negatives_to_sample_from, num_sampled, biased=biased_mode):
            triples.append((qid, pos_pid, neg_pid))

    return triples


def main(args):
    """主函数，负责执行完整的三元组生成流程。"""
    try: # 尝试加载带 5 列（包括 label）的排序文件
        rankings = load_ranking(args.ranking, types=[int, int, int, float, int])
    except: # 如果失败，尝试加载 4 列的文件（假设最后一列是 label）
        rankings = load_ranking(args.ranking, types=[int, int, int, int])

    qid2rankings = groupby_first_item(rankings)

    all_triples = []
    
    for processing_idx, qid in enumerate(qid2rankings):
        triples_for_qid = sample_for_query(qid, qid2rankings[qid], args.positives, args.depth, args.permissive, args.biased)
        all_triples.extend(triples_for_qid)
        # ... (日志打印逻辑) ...

    # ... (下采样、打乱、保存的逻辑，与 self_training.py 类似) ...
    # ... (此处省略以保持简洁) ...
    
    with open(args.output, 'w') as f:
        for example in all_triples:
            ujson.dump(example, f)
            f.write('\n')

    save_metadata(f'{args.output}.meta', args)
    print_message(f"\n#> 已生成三元组文件: {args.output}")
    print_message("#> 完成。")


if __name__ == "__main__":
    parser = ArgumentParser(description='从带标注的排序列表中创建训练三元组。')
    # ... (命令行参数定义) ...
    parser.add_argument('--ranking', required=True, help="输入的带标注排序文件路径。")
    parser.add_argument('--output', required=True, help="输出的三元组文件路径。")
    parser.add_argument('--positives', required=True, nargs='+', help="正例选择策略，格式如 '5,50' (最多5个，排名前50)")
    parser.add_argument('--depth', required=True, type=int, help="负例采样的最大深度。")
    parser.add_argument('--permissive', action='store_true', help="是否启用 permissive 模式。")
    parser.add_argument('--biased', type=int, choices=[100, 200], default=None, help="启用难负例偏置采样。")
    parser.add_argument('--seed', default=12345, type=int, help="随机种子。")
    args = parser.parse_args()
    
    random.seed(args.seed)
    args.positives = [list(map(int, config.split(','))) for config in args.positives]
    
    assert not os.path.exists(args.output), f"输出文件 {args.output} 已存在！"
    create_directory(os.path.dirname(args.output))
    
    main(args)