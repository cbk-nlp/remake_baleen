# 文件名: utility/rankings/merge.py

import os
import tqdm
from argparse import ArgumentParser
from collections import defaultdict

from colbert.utils.utils import print_message, file_tqdm


def main(args):
    """主函数，负责加载、合并、重排序和保存。"""
    # 使用一个字典来存储所有查询的排序结果，{qid: [(score, rank, pid), ...]}
    all_rankings = defaultdict(list)

    # --- 1. 加载所有输入的排序文件 ---
    for path in args.input:
        print_message(f"#> 正在从 {path} 加载排序结果...")
        with open(path) as f:
            for line in file_tqdm(f):
                try:
                    qid, pid, rank, score = line.strip().split('\t')
                    qid, pid, rank, score = int(qid), int(pid), int(rank), float(score)
                    all_rankings[qid].append((score, rank, pid))
                except (ValueError, IndexError):
                    continue

    # --- 2. 合并、重排序并写入新文件 ---
    with open(args.output, 'w') as f:
        print_message(f"#> 正在将合并后的排序结果写入 {args.output} ...")
        
        # 遍历每个查询
        for qid in tqdm.tqdm(all_rankings.keys(), desc="合并排序结果"):
            # 对该查询的所有结果按分数降序排序
            merged_ranking = sorted(all_rankings[qid], reverse=True)

            # 遍历排序后的结果，并分配新的 rank
            for new_rank, (score, original_rank, pid) in enumerate(merged_ranking, 1):
                # 如果设置了深度限制，则只保留 top-k
                if args.depth > 0 and new_rank > args.depth:
                    break
                
                line = f"{qid}\t{pid}\t{new_rank}\t{score}\n"
                f.write(line)

    print_message("#> 完成！")


if __name__ == "__main__":
    parser = ArgumentParser(description="合并多个排序文件，并根据分数重新排序。")
    parser.add_argument('--input', dest='input', required=True, nargs='+', help="一个或多个输入的排序文件路径")
    parser.add_argument('--output', dest='output', required=True, type=str, help="合并后的输出文件路径")
    parser.add_argument('--depth', dest='depth', required=True, type=int, help="合并后每个查询保留的最大深度 (top-k)。设为 -1 表示保留所有。")
    args = parser.parse_args()

    assert not os.path.exists(args.output), f"输出文件 {args.output} 已存在！"

    main(args)