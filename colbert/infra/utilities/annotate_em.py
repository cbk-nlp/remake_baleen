# 文件名: colbert/infra/utilities/annotate_em.py

from multiprocessing import Pool

# 导入 colbert 内部模块
from colbert.infra.run import Run
from colbert.data.collection import Collection
from colbert.data.ranking import Ranking
from colbert.utils.utils import groupby_first_item, print_message

# 导入 utility 模块
from utility.utils.qa_loaders import load_qas_
from utility.utils.save_metadata import format_metadata
from utility.evaluate.annotate_EM_helpers import tokenize_all_answers, assign_label_to_passage, check_sizes


class AnnotateEM:
    """
    一个用于自动进行精确匹配（Exact Match, EM）标注的工具类。

    它接收一个检索结果的排序列表（ranking）和一份包含标准答案的问答（QA）数据，
    然后为排序列表中的每个段落判断它是否包含了标准答案中的任意一个。
    最终，它会输出一个带有标注（label，1表示包含答案，0表示不包含）的新排序列表。
    """

    def __init__(self, collection, qas):
        """
        初始化 AnnotateEM。

        Args:
            collection (str or Collection): 文档集合的路径或对象。
            qas (str): 问答（QA）数据文件的路径。
        """
        qas_data = load_qas_(qas)
        self.collection = Collection.cast(collection)
        self.parallel_pool = Pool(30)

        print_message('#> 正在并行地对标准答案进行分词...')
        tokenized_qas = list(self.parallel_pool.map(tokenize_all_answers, qas_data))

        self.qid2answers = {qid: tok_answers for qid, _, tok_answers in tokenized_qas}
        assert len(qas_data) == len(self.qid2answers)

    def annotate(self, ranking):
        """
        对给定的排序列表进行标注。

        Args:
            ranking (str or Ranking): 排序列表的路径或对象。

        Returns:
            Ranking: 一个新的、包含了 EM 标注的 Ranking 对象。
        """
        rankings = Ranking.cast(ranking)

        print_message('#> 正在从 PID 查找段落内容...')
        expanded_rankings = [
            (qid, pid, rank, self.collection[pid], self.qid2answers.get(qid, []))
            for qid, pid, rank, *_ in rankings.tolist()
            if qid in self.qid2answers # 只处理有标准答案的查询
        ]

        print_message('#> 正在并行地分配标签...')
        labeled_rankings = list(self.parallel_pool.map(assign_label_to_passage, enumerate(expanded_rankings)))

        self.qid2rankings = groupby_first_item(labeled_rankings)
        self.num_judged_queries, self.num_ranked_queries = check_sizes(self.qid2answers, self.qid2rankings)
        self.success, self.counts = self._compute_labels(self.qid2answers, self.qid2rankings)

        # 创建一个新的 Ranking 对象，并附加上下文来源信息
        return Ranking(data=self.qid2rankings, provenance=("AnnotateEM", rankings.provenance()))

    def _compute_labels(self, qid2answers, qid2rankings):
        """在内部计算成功率等指标。"""
        cutoffs = [1, 5, 10, 20, 30, 50, 100, 1000, 'all']
        success = {cutoff: 0.0 for cutoff in cutoffs}
        counts = {cutoff: 0.0 for cutoff in cutoffs}

        for qid in qid2answers:
            if qid not in qid2rankings:
                continue
            
            labels = [label for _, _, label in qid2rankings[qid]]
            
            for cutoff in cutoffs:
                cutoff_val = cutoff if cutoff != 'all' else len(labels)
                if sum(labels[:cutoff_val]) > 0:
                    success[cutoff] += 1.0
                counts[cutoff] += sum(labels[:cutoff_val])

        return success, counts

    def save(self, new_path):
        """保存标注后的排序列表和计算出的评估指标。"""
        print_message("#> 正在将输出转储至", new_path, "...")
        Ranking(data=self.qid2rankings).save(new_path)

        # 保存评估指标到 .metrics 文件
        with Run().open(f'{new_path}.metrics', 'w') as f:
            d = {'num_ranked_queries': self.num_ranked_queries, 'num_judged_queries': self.num_judged_queries}
            extra = '__WARNING' if self.num_judged_queries != self.num_ranked_queries else ''
            d[f'success{extra}'] = {k: v / self.num_judged_queries for k, v in self.success.items()}
            d[f'counts{extra}'] = {k: v / self.num_judged_queries for k, v in self.counts.items()}
            f.write(format_metadata(d) + '\n')