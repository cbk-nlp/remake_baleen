# 文件名: utility/utils/qa_loaders.py

import ujson

from colbert.utils.utils import print_message, file_tqdm


def load_collection_(path, retain_titles=True):
    """
    从一个 TSV 文件加载文档集合。

    Args:
        path (str): 集合文件的路径。
        retain_titles (bool, optional): 如果为 True，并且文件包含标题，
                                       则将标题和段落内容拼接成 "title | passage" 的格式。
                                       默认为 True。
    
    Returns:
        list[str]: 文档（段落）内容的列表。
    """
    collection = []
    print_message(f"#> 正在从 {path} 加载文档集合...")
    with open(path) as f:
        for line in file_tqdm(f):
            try:
                # 假设格式为 pid\tpassage\ttitle
                _, passage, title = line.strip().split('\t')
                if retain_titles and title:
                    passage = f"{title} | {passage}"
                collection.append(passage)
            except ValueError:
                # 兼容 pid\tpassage 的格式
                _, passage = line.strip().split('\t')
                collection.append(passage)
    return collection


def load_qas_(path):
    """
    从一个 JSON-Lines 文件加载问答（QA）数据。
    文件中的每一行都是一个独立的 JSON 对象。

    Args:
        path (str): 问答数据文件的路径。

    Returns:
        list[tuple]: 一个列表，每个元素是一个元组 `(qid, question, answers_list)`。
    """
    print_message("#> 正在从", path, "加载参考问答数据...")
    triples = []
    with open(path) as f:
        for line in f:
            qa = ujson.loads(line)
            # 提取所需字段
            triples.append((qa['qid'], qa['question'], qa['answers']))
    return triples