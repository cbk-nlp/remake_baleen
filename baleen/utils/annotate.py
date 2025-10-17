# 文件名: baleen/utils/annotate.py

import os
import ujson

from colbert.utils.utils import print_message, file_tqdm


def annotate_to_file(qas_path, ranking_path):
    """
    为一个排序文件（ranking_path）添加标注。

    它会读取一个问答文件（qas_path）以获取每个查询的正确答案（support_pids），
    然后遍历排序文件，为每一行判断其 PID 是否在正确答案列表中，并添加一个
    标签（1 或 0）。

    Args:
        qas_path (str): 问答文件路径 (.json 格式)。
        ranking_path (str): 待标注的排序文件路径 (.tsv 格式)。

    Returns:
        str: 生成的带标注的新排序文件的路径。
    """
    output_path = f'{ranking_path}.annotated'
    assert not os.path.exists(output_path), f"输出文件 {output_path} 已存在！"

    # 1. 加载所有查询的正确答案 PID
    QID2pids = {}
    print_message(f"#> 正在从 {qas_path} 读取问答数据...")
    with open(qas_path) as f:
        for line in file_tqdm(f):
            example = ujson.loads(line)
            QID2pids[example['qid']] = set(example['support_pids'])

    # 2. 遍历排序文件并添加标签
    print_message(f"#> 正在从 {ranking_path} 读取排序列表并进行标注...")
    with open(ranking_path) as f_in, open(output_path, 'w') as f_out:
        for line in file_tqdm(f_in):
            qid_str, pid_str, *other = line.strip().split('\t')
            qid, pid = int(qid_str), int(pid_str)

            # 判断 pid 是否为正例
            label = 1 if pid in QID2pids.get(qid, set()) else 0
            
            # 写入新行
            f_out.write('\t'.join([qid_str, pid_str] + other + [str(label)]) + '\n')

    print_message(f"#> 已生成带标注的文件: {output_path}")
    print_message("#> 完成！")
    return output_path