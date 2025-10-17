# 文件名: utility/evaluate/annotate_EM.py

import os
import random
from argparse import ArgumentParser
from multiprocessing import Pool

# 导入 ColBERT 和 utility 中的辅助函数
from colbert.utils.utils import print_message, load_ranking, groupby_first_item
from utility.utils.qa_loaders import load_qas_, load_collection_
from utility.utils.save_metadata import format_metadata, get_metadata
# 导入 EM 标注的核心辅助函数
from utility.evaluate.annotate_EM_helpers import (
    tokenize_all_answers,
    assign_label_to_passage,
    check_sizes,
    compute_and_write_labels
)


def main(args):
    """
    主函数，执行完整的 EM 自动标注和评估流程。

    流程如下:
    1.  加载问答数据 (QAs)、文档集合 (collection) 和模型的排序结果 (ranking)。
    2.  使用多进程并行地对标准答案进行分词和标准化处理。
    3.  将排序结果与文档内容、标准答案关联起来。
    4.  使用多进程并行地为每个检索到的段落分配 EM 标签（1 表示包含答案，0 表示不包含）。
    5.  将带有标签的排序结果写入新的文件。
    6.  计算并保存评估指标（如 Success Rate）到另一个文件中。
    """
    # 加载所需数据
    qas = load_qas_(args.qas)
    collection = load_collection_(args.collection, retain_titles=True)
    rankings = load_ranking(args.ranking)
    
    # 初始化一个进程池以进行并行处理
    parallel_pool = Pool(30)

    print_message('#> 正在并行地对标准答案进行分词...')
    # 对每个问答对中的答案进行标准化分词
    qas = list(parallel_pool.map(tokenize_all_answers, qas))
    # 创建一个从 qid 到标准化答案列表的映射
    qid2answers = {qid: tok_answers for qid, _, tok_answers in qas}
    assert len(qas) == len(qid2answers)

    print_message('#> 正在从 PID 查找段落内容...')
    # 将排序列表中的每一项与段落内容和对应的标准答案关联起来
    expanded_rankings = [
        (qid, pid, rank, collection[pid], qid2answers[qid])
        for qid, pid, rank, *_ in rankings
        if qid in qid2answers # 只处理有标准答案的查询
    ]

    print_message('#> 正在并行地分配标签...')
    # 对每个 (段落, 答案) 对进行 EM 判断
    labeled_rankings = list(parallel_pool.map(assign_label_to_passage, enumerate(expanded_rankings)))

    print_message("#> 正在将输出转储至", args.output, "...")
    # 按 qid 对标注后的结果进行分组
    qid2rankings = groupby_first_item(labeled_rankings)

    # 检查查询数量是否一致
    num_judged_queries, num_ranked_queries = check_sizes(qid2answers, qid2rankings)

    # 计算指标并将带标签的排序结果写入文件
    success, counts = compute_and_write_labels(args.output, qid2answers, qid2rankings)

    # 将最终的评估指标保存到 .metrics 文件中
    with open(args.output_metrics, 'w') as f:
        d = {'num_ranked_queries': num_ranked_queries, 'num_judged_queries': num_judged_queries}
        # 如果参与评估的查询数量和有排序结果的查询数量不一致，则添加警告标记
        extra = '__WARNING' if num_judged_queries != num_ranked_queries else ''
        d[f'success{extra}'] = {k: v / num_judged_queries for k, v in success.items()}
        d[f'counts{extra}'] = {k: v / num_judged_queries for k, v in counts.items()}
        d['arguments'] = get_metadata(args)
        f.write(format_metadata(d) + '\n')

    print('\n#>\n#> 已生成标注文件:', args.output)
    print('#> 已生成指标文件:', args.output_metrics)
    print("#> 完成\n")


if __name__ == "__main__":
    random.seed(12345)

    # --- 命令行参数解析 ---
    parser = ArgumentParser(description='对排序列表进行精确匹配 (EM) 自动标注。')
    parser.add_argument('--qas', dest='qas', required=True, type=str, help="包含标准答案的问答 (QA) 文件路径 (.json 格式)")
    parser.add_argument('--collection', dest='collection', required=True, type=str, help="文档集合文件路径 (.tsv 格式)")
    parser.add_argument('--ranking', dest='ranking', required=True, type=str, help="待标注的模型排序结果文件路径 (.tsv 格式)")
    args = parser.parse_args()

    # 自动生成输出文件名
    args.output = f'{args.ranking}.annotated'
    args.output_metrics = f'{args.ranking}.annotated.metrics'

    # 确保不会覆盖已存在的文件
    assert not os.path.exists(args.output), f"输出文件 {args.output} 已存在！"

    main(args)