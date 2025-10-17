# 文件名: utility/evaluate/annotate_EM_helpers.py

from colbert.utils.utils import print_message
# 导入 DPR 项目中用于文本标准化和答案匹配的函数
from utility.utils.dpr import DPR_normalize, has_answer


def tokenize_all_answers(args):
    """
    一个辅助函数，用于对单个问答对 (QA pair) 中的所有答案进行标准化处理。
    通常被 `multiprocessing.Pool.map` 调用以实现并行处理。

    Args:
        args (tuple): 一个元组，包含 (qid, question, answers_list)。

    Returns:
        tuple: (qid, question, normalized_answers_list)。
    """
    qid, question, answers = args
    # 使用 DPR_normalize 函数对每个答案进行小写化、去标点等标准化操作
    return qid, question, [DPR_normalize(ans) for ans in answers]


def assign_label_to_passage(args):
    """
    为单个段落分配精确匹配 (EM) 标签。
    它会检查给定的段落文本中是否包含任何一个标准答案。

    Args:
        args (tuple): 一个元组，包含 (索引, (qid, pid, rank, passage_text, tokenized_answers))。

    Returns:
        tuple: (qid, pid, rank, label)，其中 label 是一个布尔值 (True/False)。
    """
    idx, (qid, pid, rank, passage, tokenized_answers) = args

    if idx % (1_000_000) == 0 and idx > 0:
        print(f"已处理 {idx // 1_000_000}M 个段落...")

    # 调用 has_answer 函数执行核心的匹配逻辑
    label = has_answer(tokenized_answers, passage)
    return qid, pid, rank, label


def check_sizes(qid2answers, qid2rankings):
    """
    检查并打印参与评估的查询数量和实际有排序结果的查询数量。
    如果两者不一致，会打印一条警告信息。
    """
    num_judged_queries = len(qid2answers)
    num_ranked_queries = len(qid2rankings)

    print_message('参与评估的查询总数 =', num_judged_queries)
    print_message('有排序结果的查询数 =', num_ranked_queries)

    if num_judged_queries != num_ranked_queries:
        assert num_ranked_queries <= num_judged_queries
        print_message('\n[警告] 参与评估的查询总数与有排序结果的查询数不一致！\n')
    
    return num_judged_queries, num_ranked_queries


def compute_and_write_labels(output_path, qid2answers, qid2rankings):
    """
    计算评估指标（如不同深度的成功率），并将带标签的排序结果写入文件。

    Args:
        output_path (str): 保存带标签排序结果的文件路径。
        qid2answers (dict): 查询ID到答案的映射。
        qid2rankings (dict): 查询ID到已标注排序结果的映射。

    Returns:
        tuple: (success, counts)，两个字典，分别记录了在不同深度下的成功次数和找到的答案总数。
    """
    # 定义要评估的深度
    cutoffs = [1, 5, 10, 20, 30, 50, 100, 1000, 'all']
    success = {cutoff: 0.0 for cutoff in cutoffs}
    counts = {cutoff: 0.0 for cutoff in cutoffs}

    with open(output_path, 'w') as f:
        # 遍历所有需要评估的查询
        for qid in qid2answers:
            if qid not in qid2rankings:
                continue

            labels = []
            # 将该查询的标注结果写入文件
            for pid, rank, label in qid2rankings[qid]:
                labels.append(label)
                line = f"{qid}\t{pid}\t{rank}\t{int(label)}\n"
                f.write(line)

            # 累积计算指标
            for cutoff in cutoffs:
                # 'all' 表示考虑所有深度的结果
                cutoff_val = cutoff if cutoff != 'all' else len(labels)
                
                # Success@k: 只要在 top-k 中找到一个答案，就记为成功
                if sum(labels[:cutoff_val]) > 0:
                    success[cutoff] += 1.0
                
                # Counts@k: 累加在 top-k 中找到的答案总数
                counts[cutoff] += sum(labels[:cutoff_val])

    return success, counts