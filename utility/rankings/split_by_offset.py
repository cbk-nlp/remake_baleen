# 文件名: utility/rankings/split_by_offset.py

import os
from argparse import ArgumentParser


def main(args):
    """主函数，执行拆分逻辑。"""
    # 根据提供的名称列表生成输出文件路径
    output_paths = [f'{args.ranking}.{name}' for name in args.names]
    assert all(not os.path.exists(path) for path in output_paths), "部分或全部输出文件已存在！"

    # 打开所有输出文件
    output_files = [open(path, 'w') for path in output_paths]

    print(f"#> 正在将 {args.ranking} 拆分为 {len(output_files)} 个文件...")
    with open(args.ranking) as f:
        for line in f:
            qid_str, pid, rank, *other = line.strip().split('\t')
            qid = int(qid_str)
            
            # --- 核心逻辑 ---
            # 根据 qid 和 gap 计算它属于哪个原始文件
            split_index = (qid - 1) // args.gap
            # 还原原始的 qid
            original_qid = qid % args.gap
            if original_qid == 0: original_qid = args.gap # 处理边界情况
            
            # 选择正确的输出文件
            target_file = output_files[split_index]
            
            # 写入还原后的行
            target_file.write('\t'.join([str(original_qid), pid, rank] + other) + '\n')
    
    # 关闭所有文件
    for f in output_files:
        f.close()
    
    print("#> 完成！")


if __name__ == "__main__":
    parser = ArgumentParser(description='根据 QID 的偏移量 (offset) 和间隔 (gap) 拆分一个合并后的排序文件。')
    parser.add_argument('--ranking', dest='ranking', required=True, help="待拆分的合并排序文件路径。")
    parser.add_argument('--names', dest='names', required=True, nargs='+', help="拆分后每个文件的名称后缀列表 (顺序必须正确！)。")
    parser.add_argument('--gap', dest='gap', required=True, type=int, help="合并时使用的 QID 间隔值。")
    args = parser.parse_args()

    main(args)