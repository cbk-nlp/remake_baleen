# 文件名: utility/supervision/self_training.py

import os
import sys
import git
import ujson
import random
from argparse import ArgumentParser

from colbert.utils.utils import print_message, load_ranking, groupby_first_item

# 定义生成三元组数量的上限
MAX_NUM_TRIPLES = 40_000_000


def sample_negatives(negatives, num_sampled):
    """从给定的负例列表中随机采样指定数量的负例。"""
    num_sampled = min(len(negatives), num_sampled)
    return random.sample(negatives, num_sampled)


def sample_for_query(qid, ranking, npositives, depth_positive, depth_negative, cutoff_negative):
    """
    为单个查询从其排序列表中采样正例和负例，以构建训练三元组。

    这是一个“自训练”的实现，它基于以下启发式规则：
    - 排名在 `depth_positive` 之内的文档被视为“正例”。
    - 排名在 `cutoff_negative` 和 `depth_negative` 之间的文档被视为“负例”。

    Args:
        qid (int): 查询ID。
        ranking (list): 该查询的排序结果列表，格式为 `[(pid, rank, ...), ...]`。
        npositives (int): 每个负例需要配对的正例数量。
        depth_positive (int): 被视为正例的最大排名。
        depth_negative (int): 考虑作为负例的最大排名。
        cutoff_negative (int): 被视为负例的最小排名。

    Returns:
        list[tuple]: 为该查询生成的三元组列表 `[(qid, positive_pid, negative_pid), ...]`。
    """
    assert npositives <= depth_positive < cutoff_negative < depth_negative

    positives, negatives, triples = [], [], []

    # 遍历排序列表，收集正例和负例
    for pid, rank, *_ in ranking:
        if rank > depth_negative:
            break
        if rank <= depth_positive:
            positives.append(pid)
        elif rank > cutoff_negative:
            negatives.append(pid)

    # 每个负例都与 npositives 个随机选择的正例配对
    num_sampled_negatives = 100
    for neg_pid in sample_negatives(negatives, num_sampled_negatives):
        # 随机选择 npositives 个正例
        sampled_pos_pids = random.sample(positives, npositives)
        # 创建三元组
        for pos_pid in sampled_pos_pids:
            triples.append((qid, pos_pid, neg_pid))

    return triples


def main(args):
    """主函数，负责执行完整的三元组生成流程。"""
    # 加载排序文件
    rankings = load_ranking(args.ranking, types=[int, int, int, float, int])
    # 按 qid 分组
    qid2rankings = groupby_first_item(rankings)

    all_triples = []
    
    # 遍历每个查询并生成三元组
    for processing_idx, qid in enumerate(qid2rankings):
        triples_for_qid = sample_for_query(qid, qid2rankings[qid], args.positives, args.depth_positive, args.depth_negative, args.cutoff_negative)
        all_triples.extend(triples_for_qid)

        if (processing_idx + 1) % 10000 == 0:
            print_message(f"#> 已处理 {processing_idx + 1} 个查询，生成了 {len(all_triples) / 1000:.1f}k 个三元组。")

    # 如果生成的三元组过多，则进行下采样
    print_message(f"#> 原始三元组数量 = {len(all_triples)}")
    if len(all_triples) > MAX_NUM_TRIPLES:
        all_triples = random.sample(all_triples, MAX_NUM_TRIPLES)

    # 打乱并保存
    print_message("#> 正在打乱三元组...")
    random.shuffle(all_triples)

    print_message(f"#> 正在将 {len(all_triples) / 1e6:.2f}M 个样本写入文件...")
    with open(args.output, 'w') as f:
        for example in all_triples:
            ujson.dump(example, f)
            f.write('\n')

    # 保存元数据
    with open(f'{args.output}.meta', 'w') as f:
        args.cmd = ' '.join(sys.argv)
        args.git_hash = git.Repo(search_parent_directories=True).head.object.hexsha
        ujson.dump(args.__dict__, f, indent=4)
        f.write('\n')

    print_message(f"\n#> 已生成三元组文件: {args.output}")
    print_message("#> 完成。")


if __name__ == "__main__":
    random.seed(12345)
    parser = ArgumentParser(description='通过自训练（self-training）从排序列表中创建训练三元组。')
    # 输入/输出参数
    parser.add_argument('--ranking', required=True, help="输入的排序文件路径。")
    parser.add_argument('--output', required=True, help="输出的三元组文件路径。")
    # 弱监督参数
    parser.add_argument('--positives', required=True, type=int, help="每个负例配对的正例数量。")
    parser.add_argument('--depth+', dest='depth_positive', required=True, type=int, help="被视为正例的最大排名。")
    parser.add_argument('--depth-', dest='depth_negative', required=True, type=int, help="考虑作为负例的最大排名。")
    parser.add_argument('--cutoff-', dest='cutoff_negative', required=True, type=int, help="被视为负例的最小排名。")
    args = parser.parse_args()

    assert not os.path.exists(args.output), f"输出文件 {args.output} 已存在！"
    main(args)