# 文件名: utility/evaluate/msmarco_passages.py

import os
from argparse import ArgumentParser
from collections import defaultdict

# 导入 colbert.utils 中的辅助函数
from colbert.utils.utils import print_message, file_tqdm


def main(args):
    """
    主函数，用于评估 MS MARCO Passages 数据集上的排序结果。

    它会计算 MRR@10 和 Recall@k (k=50, 200, 1000) 等标准指标。
    同时，它还提供一个可选的功能，可以将评估过程中的标签（是否为正例）
    写回到一个新的排序文件中。
    """
    qid2positives = defaultdict(list) # {qid: [positive_pid_1, ...]}
    qid2ranking = defaultdict(list)   # {qid: [(rank, pid, score), ...]}
    
    # --- 1. 加载真实标签 (QRELs) ---
    with open(args.qrels) as f:
        print_message(f"#> 正在从 {args.qrels} 加载 QRELs...")
        for line in file_tqdm(f):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1, "QRELs 文件中的标签应为 1"
            qid2positives[qid].append(pid)

    # --- 2. 加载模型排序结果 ---
    with open(args.ranking) as f:
        print_message(f"#> 正在从 {args.ranking} 加载排序列表...")
        for line in file_tqdm(f):
            qid, pid, rank, *score_str = line.strip().split('\t')
            qid, pid, rank = int(qid), int(pid), int(rank)
            score = float(score_str[0]) if score_str else None
            qid2ranking[qid].append((rank, pid, score))

    # 检查查询集合是否一致
    assert set(qid2ranking.keys()).issubset(set(qid2positives.keys())), "排序文件中的查询ID必须是QRELs文件查询ID的子集"

    num_judged_queries = len(qid2positives)
    num_ranked_queries = len(qid2ranking)

    if num_judged_queries != num_ranked_queries:
        print_message("\n[警告] QRELs 中的查询数量与排序文件中的查询数量不一致！"
                      f"({num_judged_queries} != {num_ranked_queries})\n")

    # --- 3. 计算评估指标 ---
    qid2mrr = {}
    recall_depths = [50, 200, 1000]
    qid2recall = {depth: defaultdict(float) for depth in recall_depths}

    print_message(f"#> 正在为 {num_judged_queries} 个查询计算 MRR@10 和 Recall...")
    for qid, positives in qid2positives.items():
        if qid not in qid2ranking:
            continue
        
        ranking = qid2ranking[qid]
        
        # 计算 MRR@10
        for rank_idx, (rank, pid, _) in enumerate(ranking):
            if pid in positives:
                if rank <= 10:
                    qid2mrr[qid] = 1.0 / rank
                # 找到第一个正例后即可停止计算 MRR
                break
        
        # 计算 Recall@k
        for rank, pid, _ in ranking:
            if pid in positives:
                for depth in recall_depths:
                    if rank <= depth:
                        qid2recall[depth][qid] += 1.0 / len(positives)

    # --- 4. 打印结果 ---
    mrr_10_sum = sum(qid2mrr.values())
    print_message(f"\n#> MRR@10 (基于所有QRELs查询) = {mrr_10_sum / num_judged_queries:.4f}")
    print_message(f"#> MRR@10 (仅基于有排序的查询) = {mrr_10_sum / num_ranked_queries:.4f}\n")

    for depth in recall_depths:
        recall_sum = sum(qid2recall[depth].values())
        print_message(f"#> Recall@{depth} (基于所有QRELs查询) = {recall_sum / num_judged_queries:.4f}")
        print_message(f"#> Recall@{depth} (仅基于有排序的查询) = {recall_sum / num_ranked_queries:.4f}\n")

    # --- 5. (可选) 生成带标注的排序文件 ---
    if args.annotate:
        print_message(f"#> 正在将带标注的结果写入 {args.output}...")
        with open(args.output, 'w') as f:
            for qid in qid2ranking:
                positives = set(qid2positives.get(qid, []))
                for rank, pid, score in qid2ranking[qid]:
                    label = 1 if pid in positives else 0
                    f.write(f"{qid}\t{pid}\t{rank}\t{score}\t{label}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="评估 MS MARCO Passages 数据集上的排序结果。")
    parser.add_argument('--qrels', dest='qrels', required=True, type=str, help="官方 QRELs 文件的路径")
    parser.add_argument('--ranking', dest='ranking', required=True, type=str, help="模型生成的排序文件路径")
    parser.add_argument('--annotate', dest='annotate', action='store_true', help="如果设置，将生成一个带标签的新排序文件")
    args = parser.parse_args()

    if args.annotate:
        args.output = f'{args.ranking}.annotated'
        assert not os.path.exists(args.output), f"输出文件 {args.output} 已存在！"

    main(args)