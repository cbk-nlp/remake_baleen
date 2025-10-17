# 文件名: baleen/utils/loaders.py

import ujson
from colbert.utils.utils import print_message


def load_contexts(first_hop_topk_path):
    """
    从一个 JSON-Lines 文件中加载上下文信息（或称为 "backgrounds"）。
    这个文件通常是多跳检索中第一跳的结果。
    """
    qid2backgrounds = {}
    print_message(f"#> 正在从 {first_hop_topk_path} 加载上下文...")
    with open(first_hop_topk_path) as f:
        for line in f:
            qid, facts = ujson.loads(line)
            # 将列表转换为元组，以便可以作为字典的键或放入集合
            facts_tuples = [tuple(f) if isinstance(f, list) else f for f in facts]
            qid2backgrounds[qid] = facts_tuples
            
    print_message(f"#> 已加载 {len(qid2backgrounds)} 个查询的上下文。")
    return qid2backgrounds


def load_collectionX(collection_path, dict_in_dict=False):
    """
    加载一个特殊的、按句子切分好的文档集合 (CollectionX)。
    这个集合的 key 通常是 `(pid, sentence_id)`。

    Args:
        collection_path (str): 集合文件的路径 (JSON-Lines 格式)。
        dict_in_dict (bool): 如果为 True，则返回嵌套字典 `{pid: {sid: text}}`，
                             否则返回扁平字典 `{(pid, sid): text}`。
    """
    print_message("#> 正在加载句子集合 (CollectionX)...")
    collectionX = {}
    with open(collection_path) as f:
        for line_idx, line in enumerate(f):
            data = ujson.loads(line)
            pid = data['pid']
            sentences = data['text']
            title = data['title']
            
            # 将标题和每个句子拼接
            full_sentences = [f"{title} | {sentence}" for sentence in sentences]

            if dict_in_dict:
                collectionX[pid] = {sid: text for sid, text in enumerate(full_sentences)}
            else:
                for sid, text in enumerate(full_sentences):
                    collectionX[(pid, sid)] = text
                    
    return collectionX