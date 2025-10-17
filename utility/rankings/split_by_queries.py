# 文件名: utility/rankings/split_by_queries.py

import os
from argparse import ArgumentParser

from colbert.utils.utils import print_message, file_tqdm


def main(args):
    """主函数，执行拆分逻辑。"""
    # --- 1. 构建 qid 到文件索引的映射 ---
    qid_to_file_idx = {}
    print_message("#> 正在构建 qid -> 文件索引 的映射...")
    for file_idx, queries_path in enumerate(args.all_queries):
        with open(queries_path) as f:
            for line in f:
                try:
                    qid, *_ = line.strip().split('\t')
                    qid = int(qid)
                    # 确保同一个 qid 不会出现在多个查询文件中
                    assert qid not in qid_to_file_idx, f"QID {qid} 在多个查询文件中重复出现！"
                    qid_to_file_idx[qid] = file_idx
                except (ValueError, IndexError):
                    continue

    # --- 2. 准备输出文件 ---
    output_paths = [f'{args.ranking}.{idx}' for idx in range(len(args.all_queries))]
    assert all(not os.path.exists(path) for path in output_paths), "部分或全部输出文件已存在！"
    output_files = [open(path, 'w') for path in output_paths]

    # --- 3. 遍历排序文件并进行拆分 ---
    print_message(f"#> 正在拆分排序文件 {args.ranking}...")
    with open(args.ranking) as f:
        for line in file_tqdm(f):
            try:
                qid_str, *_ = line.strip().split('\t')
                qid = int(qid_str)
                
                # 找到该 qid 对应的文件索引
                target_file_idx = qid_to_file_idx.get(qid)
                
                # 如果找到了，就将该行写入对应的输出文件
                if target_file_idx is not None:
                    output_files[target_file_idx].write(line)
            except (ValueError, IndexError):
                continue
    
    # 关闭所有输出文件
    for f in output_files:
        f.close()

    print("\n#> 已生成以下文件:")
    for path in output_paths:
        print(f"#> - {path}")
    print("#> 完成！")


if __name__ == "__main__":
    parser = ArgumentParser(description='根据一个或多个查询文件，将一个大的排序结果文件拆分成对应的部分。')
    parser.add_argument('--ranking', dest='ranking', required=True, type=str, help="待拆分的完整排序文件路径。")
    parser.add_argument('--all-queries', dest='all_queries', required=True, nargs='+', type=str, help="一个或多个查询文件的路径列表，用于决定拆分规则。")
    args = parser.parse_args()

    main(args)