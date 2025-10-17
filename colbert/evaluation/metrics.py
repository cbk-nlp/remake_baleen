# 文件名: colbert/evaluation/metrics.py

import ujson
from collections import defaultdict

# 导入 Run 类，用于日志记录
from colbert.infra.run import Run


class Metrics:
    """
    一个用于计算、记录和保存信息检索评估指标的类。
    支持的指标包括 MRR, Recall, 和 Success Rate。
    """

    def __init__(self, mrr_depths: set, recall_depths: set, success_depths: set, total_queries=None):
        """
        初始化 Metrics 对象。

        Args:
            mrr_depths (set): 计算 MRR (Mean Reciprocal Rank) 的深度集合，例如 {10, 100}。
            recall_depths (set): 计算 Recall 的深度集合。
            success_depths (set): 计算 Success Rate 的深度集合。
            total_queries (int, optional): 查询总数，用于最终的验证。
        """
        self.results = {}
        self.mrr_sums = {depth: 0.0 for depth in mrr_depths}
        self.recall_sums = {depth: 0.0 for depth in recall_depths}
        self.success_sums = {depth: 0.0 for depth in success_depths}
        self.total_queries = total_queries
        self.num_queries_added = 0

    def add(self, query_idx, query_key, ranking, gold_positives):
        """
        为单个查询添加排序结果和真实标签，并累积指标。

        Args:
            query_idx (int): 查询的索引。
            query_key (any): 查询的唯一标识符 (例如 qid)。
            ranking (list): 模型的排序结果，格式为 [(rank, pid, score), ...]。
            gold_positives (list or set): 该查询的所有相关文档的 pid 列表。
        """
        self.num_queries_added += 1
        assert query_key not in self.results, f"查询 {query_key} 的结果已被添加过。"
        assert len(set(gold_positives)) == len(gold_positives), "真实正例中存在重复。"
        assert len(set([pid for _, pid, _ in ranking])) == len(ranking), "排序结果中存在重复的 pid。"

        self.results[query_key] = ranking
        
        # 找到所有正例在排序列表中的位置（从 0 开始）
        positives = [i for i, (_, pid, _) in enumerate(ranking) if pid in gold_positives]

        # 如果没有找到任何正例，则直接返回
        if len(positives) == 0:
            return

        first_positive_rank = positives[0] + 1.0  # 排名从 1 开始

        # 计算 MRR
        for depth in self.mrr_sums:
            if first_positive_rank <= depth:
                self.mrr_sums[depth] += 1.0 / first_positive_rank
        
        # 计算 Success Rate
        for depth in self.success_sums:
            if first_positive_rank <= depth:
                self.success_sums[depth] += 1.0

        # 计算 Recall
        for depth in self.recall_sums:
            num_positives_up_to_depth = len([pos for pos in positives if pos < depth])
            self.recall_sums[depth] += num_positives_up_to_depth / len(gold_positives)

    def print_metrics(self, num_queries):
        """打印当前计算出的平均指标。"""
        print("\n--- 评估指标 ---")
        for depth in sorted(self.mrr_sums):
            print(f"MRR@{depth} = {self.mrr_sums[depth] / num_queries:.4f}")
        for depth in sorted(self.success_sums):
            print(f"Success@{depth} = {self.success_sums[depth] / num_queries:.4f}")
        for depth in sorted(self.recall_sums):
            print(f"Recall@{depth} = {self.recall_sums[depth] / num_queries:.4f}")
        print("------------------\n")

    def log(self, query_idx):
        """将指标记录到实验日志中 (例如，用于 TensorBoard 或 MLflow)。"""
        num_queries = query_idx + 1
        Run.log_metric("ranking/num_queries_added", self.num_queries_added, query_idx)

        for depth in sorted(self.mrr_sums):
            score = self.mrr_sums[depth] / num_queries
            Run.log_metric(f"ranking/MRR@{depth}", score, query_idx)
        for depth in sorted(self.success_sums):
            score = self.success_sums[depth] / num_queries
            Run.log_metric(f"ranking/Success@{depth}", score, query_idx)
        for depth in sorted(self.recall_sums):
            score = self.recall_sums[depth] / num_queries
            Run.log_metric(f"ranking/Recall@{depth}", score, query_idx)

    def output_final_metrics(self, path, num_queries):
        """
        计算最终指标，打印并保存到 JSON 文件。

        Args:
            path (str): 保存指标的输出文件路径。
            num_queries (int): 用于计算平均值的查询总数。
        """
        if self.total_queries is not None:
            assert num_queries == self.total_queries, f"查询数量不匹配: {num_queries} != {self.total_queries}"

        self.print_metrics(num_queries)

        output = defaultdict(dict)
        for depth in sorted(self.mrr_sums):
            output['mrr'][depth] = self.mrr_sums[depth] / num_queries
        for depth in sorted(self.success_sums):
            output['success'][depth] = self.success_sums[depth] / num_queries
        for depth in sorted(self.recall_sums):
            output['recall'][depth] = self.recall_sums[depth] / num_queries

        with open(path, 'w') as f:
            ujson.dump(output, f, indent=4)
            f.write('\n')


def evaluate_recall(qrels, queries, topK_pids):
    """
    一个独立的函数，用于评估给定 top-K 结果的召回率。

    Args:
        qrels (dict): {qid: [positive_pids]}
        queries (dict): {qid: query_text}
        topK_pids (dict): {qid: [retrieved_pids]}
    """
    if qrels is None:
        return

    assert set(qrels.keys()) == set(queries.keys()), "qrels 和 queries 的 qid 集合不匹配。"
    
    # 计算每个查询的召回率，然后求平均
    recall_at_k = [len(set.intersection(set(qrels[qid]), set(topK_pids[qid]))) / max(1.0, len(qrels[qid]))
                   for qid in qrels]
    recall_at_k = sum(recall_at_k) / len(qrels)
    
    print(f"最大深度的召回率 (Recall @ K_max) = {recall_at_k:.3f}")