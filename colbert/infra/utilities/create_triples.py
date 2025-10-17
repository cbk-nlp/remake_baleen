# 文件名: colbert/infra/utilities/create_triples.py

import random

# 导入 colbert 内部模块
from colbert.utils.utils import print_message
from colbert.data.ranking import Ranking
from colbert.data.examples import Examples

# 导入 utility 模块
from utility.supervision.triples import sample_for_query

# 定义生成三元组数量的上限
MAX_NUM_TRIPLES = 40_000_000


class Triples:
    """
    一个用于从排序列表（ranking list）生成训练三元组的工具类。

    训练三元组是 (query, positive_passage, negative_passage) 的形式，
    是训练 ColBERT 模型所需的核心数据。这个类通过对给定的排序结果
    进行采样来自动地构建这些三元组。
    """

    def __init__(self, ranking, seed=12345):
        """
        初始化 Triples 生成器。

        Args:
            ranking (str or Ranking): 包含（带标注的）排序结果的路径或 Ranking 对象。
            seed (int, optional): 随机种子，用于保证采样过程的可复现性。
        """
        random.seed(seed)
        self.qid2rankings = Ranking.cast(ranking).todict()

    def create(self, positives, depth):
        """
        根据指定的策略创建三元组。

        Args:
            positives (list[tuple[int, int]]): 一个列表，定义了如何选择正例。
                                               每个元组 (maxBest, maxDepth) 表示
                                               “从排名前 maxDepth 的段落中，最多选择 maxBest 个正例”。
            depth (int): 负例采样的最大深度。排名低于此深度的段落才会被视为负例。

        Returns:
            Examples: 包含生成的三元组的 Examples 对象。
        """
        assert all(len(x) == 2 for x in positives)
        assert all(maxBest <= maxDepth for maxBest, maxDepth in positives), positives

        all_triples = []
        non_empty_qids = 0

        # 遍历每个查询的排序结果
        for processing_idx, qid in enumerate(self.qid2rankings):
            # 调用核心采样逻辑
            triples_for_qid = sample_for_query(qid, self.qid2rankings[qid], positives, depth, False, None)
            if triples_for_qid:
                non_empty_qids += 1
            all_triples.extend(triples_for_qid)

            if (processing_idx + 1) % 10_000 == 0:
                print_message(f"#> 已处理 {processing_idx+1} 个查询，生成了 "
                              f"{len(all_triples) / 1000:.1f}k 个三元组，来自 {non_empty_qids} 个有效查询。")
        
        print_message(f"#> 原始三元组数量 = {len(all_triples)}")
        # 如果生成的三元组过多，则进行下采样
        if len(all_triples) > MAX_NUM_TRIPLES:
            print_message(f"#> 下采样至 {MAX_NUM_TRIPLES} 个三元组...")
            all_triples = random.sample(all_triples, MAX_NUM_TRIPLES)

        print_message("#> 正在打乱三元组...")
        random.shuffle(all_triples)

        # 将结果封装在 Examples 对象中
        self.Triples = Examples(data=all_triples)
        return self.Triples