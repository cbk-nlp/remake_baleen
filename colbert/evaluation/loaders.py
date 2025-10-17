# 文件名: colbert/evaluation/loaders.py

import ujson
from collections import defaultdict, OrderedDict

# 导入工具函数和类
from colbert.utils.utils import print_message
from colbert.infra.run import Run
from colbert.evaluation.load_model import load_model


def load_queries(queries_path):
    """
    从 .tsv 文件中加载查询。

    文件格式应为: qid\tquery_text\n

    Args:
        queries_path (str): 查询文件的路径。

    Returns:
        OrderedDict: 一个有序字典，键是 qid (int)，值是 query_text (str)。
    """
    queries = OrderedDict()
    print_message("#> 正在从", queries_path, "加载查询...")

    with open(queries_path) as f:
        for line in f:
            # 解析每一行，允许行尾有额外（不使用）的列
            qid, query, *_ = line.strip().split('\t')
            qid = int(qid)
            assert qid not in queries, f"查询 QID {qid} 重复!"
            queries[qid] = query

    print_message("#> 已加载", len(queries), "个查询。所有 QID 都是唯一的。\n")
    return queries


def load_qrels(qrels_path):
    """
    从查询相关性判断（qrels）文件中加载真实标签。

    文件格式应为: qid\tQ0\tpid\tlabel\n (其中 Q0 通常是 0, label 是 1)

    Args:
        qrels_path (str): qrels 文件的路径。

    Returns:
        OrderedDict or None: 一个有序字典，键是 qid (int)，值是相关文档 pid 的列表 (list[int])。
                             如果 qrels_path 为 None，则返回 None。
    """
    if qrels_path is None:
        return None

    print_message("#> 正在从", qrels_path, "加载 qrels...")
    qrels = OrderedDict()
    with open(qrels_path, mode='r', encoding="utf-8") as f:
        for line in f:
            qid, x, pid, y = map(int, line.strip().split('\t'))
            # 标准的 qrels 格式中断言 x 为 0, y 为 1
            assert x == 0 and y == 1
            if qid not in qrels:
                qrels[qid] = []
            qrels[qid].append(pid)

    # 对每个查询的相关文档列表去重
    for qid in qrels:
        qrels[qid] = list(set(qrels[qid]))

    avg_positive = round(sum(len(qrels[qid]) for qid in qrels) / len(qrels), 2)
    print_message("#> 已加载", len(qrels), "个唯一查询的 qrels，平均每个查询有", avg_positive, "个正例。\n")
    return qrels


def load_topK(topK_path):
    """
    从一个包含 top-k 检索结果的文件中加载数据。

    文件格式应为: qid\tpid\tquery\tpassage\n

    Args:
        topK_path (str): top-k 文件的路径。

    Returns:
        tuple:
            - queries (OrderedDict): {qid: query_text}
            - topK_docs (OrderedDict): {qid: [passage_text_1, passage_text_2, ...]}
            - topK_pids (OrderedDict): {qid: [pid_1, pid_2, ...]}
    """
    queries = OrderedDict()
    topK_docs = OrderedDict()
    topK_pids = OrderedDict()

    print_message("#> 正在从", topK_path, "加载每个查询的 top-k 结果...")

    with open(topK_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx and line_idx % (10*1000*1000) == 0:
                print(line_idx, end=' ', flush=True)

            qid, pid, query, passage = line.split('\t')
            qid, pid = int(qid), int(pid)

            # 确保同一个 qid 对应的查询文本是一致的
            assert (qid not in queries) or (queries[qid] == query)
            queries[qid] = query
            if qid not in topK_docs:
                topK_docs[qid] = []
                topK_pids[qid] = []
            topK_docs[qid].append(passage)
            topK_pids[qid].append(pid)
        print()

    # 检查每个查询的 top-k pid 列表是否唯一
    assert all(len(topK_pids[qid]) == len(set(topK_pids[qid])) for qid in topK_pids)
    
    Ks = [len(topK_pids[qid]) for qid in topK_pids]
    print_message("#> max(Ks) =", max(Ks), ", avg(Ks) =", round(sum(Ks) / len(Ks), 2))
    print_message("#> 已加载", len(queries), "个唯一查询的 top-k 结果。\n")

    return queries, topK_docs, topK_pids


def load_topK_pids(topK_path, qrels):
    """
    从一个 top-k 文件中仅加载段落 ID (pid)。
    如果文件包含标注信息，也会加载正例。

    文件格式应为: qid\tpid\t[rank]\t[score]\t[label]\n
    """
    topK_pids = defaultdict(list)
    topK_positives = defaultdict(list)

    print_message("#> 正在从", topK_path, "加载每个查询的 top-k PIDs...")

    with open(topK_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx and line_idx % (10*1000*1000) == 0:
                print(line_idx, end=' ', flush=True)

            qid, pid, *rest = line.strip().split('\t')
            qid, pid = int(qid), int(pid)

            topK_pids[qid].append(pid)

            # 如果行中包含标注信息 (label)
            if len(rest) > 1:
                *_, label = rest
                if int(label) >= 1:
                    topK_positives[qid].append(pid)
        print()

    # 检查 pid 和 positive pid 列表的唯一性
    assert all(len(topK_pids[qid]) == len(set(topK_pids[qid])) for qid in topK_pids)
    assert all(len(topK_positives[qid]) == len(set(topK_positives[qid])) for qid in topK_positives)

    # 将正例列表转换为集合以提高查找效率
    topK_positives = {qid: set(topK_positives[qid]) for qid in topK_positives}
    
    Ks = [len(topK_pids[qid]) for qid in topK_pids]
    print_message("#> max(Ks) =", max(Ks), ", avg(Ks) =", round(sum(Ks) / len(Ks), 2))
    print_message("#> 已加载", len(topK_pids), "个唯一查询的 top-k PIDs。\n")

    # 处理正例数据
    if len(topK_positives) == 0:
        topK_positives = None
    else:
        assert len(topK_pids) >= len(topK_positives)
        # 确保所有在 topK_pids 中的查询在 topK_positives 中都有一个条目（即使是空列表）
        for qid in set.difference(set(topK_pids.keys()), set(topK_positives.keys())):
            topK_positives[qid] = set()
        assert len(topK_pids) == len(topK_positives)
        avg_positive = round(sum(len(topK_positives[qid]) for qid in topK_positives) / len(topK_pids), 2)
        print_message("#> 同时加载了", len(topK_positives), "个唯一查询的标注，平均每个查询有", avg_positive, "个正例。\n")

    # qrels 和标注的 top-k 文件不能同时提供
    assert qrels is None or topK_positives is None, "不能同时提供 qrels 和带标注的 top-K 文件！"
    
    # 如果 topK_positives 未从文件中加载，则使用 qrels 作为正例
    if topK_positives is None:
        topK_positives = qrels

    return topK_pids, topK_positives


def load_collection(collection_path):
    """
    从 .tsv 文件加载文档集合。

    文件格式应为: pid\tpassage_text\t[title]\n
    """
    print_message("#> 正在加载文档集合...")
    collection = []

    with open(collection_path) as f:
        for line_idx, line in enumerate(f):
            if line_idx % (1000*1000) == 0:
                print(f'{line_idx // 1000 // 1000}M', end=' ', flush=True)

            pid, passage, *rest = line.strip('\n\r ').split('\t')
            assert pid == 'id' or int(pid) == line_idx, f"PID {pid} 与行号 {line_idx} 不匹配"

            # 如果存在标题，将其与段落内容拼接
            if len(rest) >= 1:
                title = rest[0]
                passage = title + ' | ' + passage
            collection.append(passage)
    print()
    return collection


def load_colbert(args, do_print=True):
    """
    加载 ColBERT 模型并检查配置参数是否一致。
    这是一个包装函数，内部调用了 load_model。
    """
    colbert, checkpoint = load_model(args, do_print)

    # 检查命令行参数与检查点中的参数是否一致，不一致时发出警告
    for k in ['query_maxlen', 'doc_maxlen', 'dim', 'similarity', 'amp']:
        if 'arguments' in checkpoint and hasattr(args, k):
            if k in checkpoint['arguments'] and checkpoint['arguments'][k] != getattr(args, k):
                a, b = checkpoint['arguments'][k], getattr(args, k)
                Run.warn(f"检查点参数 '{k}' ({a}) 与当前参数 ({b}) 不一致")

    # 打印检查点中的参数
    if 'arguments' in checkpoint:
        if args.rank < 1:
            print(ujson.dumps(checkpoint['arguments'], indent=4))

    if do_print:
        print('\n')
    return colbert, checkpoint