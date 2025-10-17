# 文件名: utility/preprocess/queries_split.py

import os
import random
from argparse import ArgumentParser
from collections import OrderedDict

from colbert.utils.utils import print_message


def main(args):
    """主函数，负责加载、分割和保存查询。"""
    random.seed(12345)

    # --- 1. 加载所有查询 ---
    queries = OrderedDict()
    print_message(f"#> 正在从 {args.input} 加载查询...")
    with open(args.input) as f:
        for line in f:
            try:
                qid, query = line.strip().split('\t')
                queries[qid] = query
            except ValueError:
                continue # 跳过格式不正确的行

    # --- 2. 执行分割 ---
    total_size = len(queries)
    holdout_size = args.holdout
    train_size = total_size - holdout_size
    assert train_size > 0 and holdout_size > 0, "分割后的两个子集都必须有内容"

    print_message(f"#> 正在将 {total_size} 个查询分割为 ({train_size}, {holdout_size}) 大小的两个子集。")
    
    # 随机选择 holdout_size 个查询作为验证集
    all_qids = list(queries.keys())
    random.shuffle(all_qids) # 先打乱
    
    holdout_qids = set(all_qids[:holdout_size])
    train_qids = all_qids[holdout_size:]

    # --- 3. 写入输出文件 ---
    output_path_train = f'{args.input}.train'
    output_path_holdout = f'{args.input}.holdout'
    assert not os.path.exists(output_path_train), f"输出文件 {output_path_train} 已存在！"
    assert not os.path.exists(output_path_holdout), f"输出文件 {output_path_holdout} 已存在！"

    print_message(f"#> 正在将分割结果写入 {output_path_train} 和 {output_path_holdout} ...")
    with open(output_path_train, 'w') as f_train, open(output_path_holdout, 'w') as f_holdout:
        for qid, query in queries.items():
            line = f"{qid}\t{query}\n"
            if qid in holdout_qids:
                f_holdout.write(line)
            else:
                f_train.write(line)

    print_message("#> 完成！")


if __name__ == "__main__":
    parser = ArgumentParser(description="将一个查询文件随机分割成训练集和验证/留出集。")
    parser.add_argument('--input', dest='input', required=True, help="输入的查询文件路径 (.tsv 格式, qid\tquery)")
    parser.add_argument('--holdout', dest='holdout', required=True, type=int, help="指定留出集 (holdout set) 的大小")
    args = parser.parse_args()
    
    main(args)