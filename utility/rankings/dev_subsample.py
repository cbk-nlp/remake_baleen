# 文件名: utility/rankings/dev_subsample.py

import os
import random
from argparse import ArgumentParser

from colbert.utils.utils import print_message, create_directory, load_ranking, groupby_first_item
from utility.utils.qa_loaders import load_qas_


def main(args):
    """主函数，负责加载数据、采样和保存。"""
    print_message("#> 正在加载所有数据...")
    # 加载用于采样的查询集 (qas)
    qas = load_qas_(args.qas)
    # 加载完整的排序结果
    rankings = load_ranking(args.ranking)
    # 按 qid 对排序结果进行分组，以便快速查找
    qid2rankings = groupby_first_item(rankings)

    print_message("#> 正在进行子采样...")
    # 从 qas 中随机选择指定数量的查询
    qas_sample = random.sample(qas, min(args.sample, len(qas)))
    # 提取这些被选中查询的 qid
    sampled_qids = {qid for qid, *_ in qas_sample}

    # 将这些 qid 对应的排序结果写入新文件
    with open(args.output, 'w') as f:
        for qid in sampled_qids:
            if qid in qid2rankings:
                for items in qid2rankings[qid]:
                    # 重新组合成 "qid\tpid\trank..." 的格式
                    line = '\t'.join(map(str, [qid] + items)) + '\n'
                    f.write(line)

    print('\n#>\n#> 已生成子采样排序文件:', args.output)
    print("#> 完成。")


if __name__ == "__main__":
    random.seed(12345)

    parser = ArgumentParser(description='从一个大的排序文件中根据一个查询子集进行子采样。')
    parser.add_argument('--qas', dest='qas', required=True, type=str, help="用于采样的查询子集文件路径 (例如 dev set 的 qas.json)")
    parser.add_argument('--ranking', dest='ranking', required=True, help="完整的排序结果文件路径")
    parser.add_argument('--output', dest='output', required=True, help="输出的子采样排序文件的路径")
    parser.add_argument('--sample', dest='sample', default=1500, type=int, help="要采样的查询数量")
    args = parser.parse_args()

    assert not os.path.exists(args.output), f"输出文件 {args.output} 已存在！"
    create_directory(os.path.dirname(args.output))

    main(args)