# 文件名: baleen/engine.py

# 导入 Baleen 的核心组件
from baleen.utils.loaders import load_collectionX
from baleen.condenser.condense import Condenser
from baleen.hop_searcher import HopSearcher


class Baleen:
    """
    Baleen 多跳问答引擎。

    这个类是整个多跳流程的 orchestrator（总控制器）。它将 `HopSearcher`
    （用于检索）和 `Condenser`（用于信息精炼）组合在一起，通过一个循环
    来迭代地执行“检索-冷凝-再检索”的步骤。
    """

    def __init__(self, collectionX_path: str, searcher: HopSearcher, condenser: Condenser):
        """
        初始化 Baleen 引擎。

        Args:
            collectionX_path (str): 用于 Condenser 的句子集合路径。
            searcher (HopSearcher): 一个 HopSearcher 实例。
            condenser (Condenser): 一个 Condenser 实例。
        """
        self.collectionX = load_collectionX(collectionX_path)
        self.searcher = searcher
        self.condenser = condenser

    def search(self, query, num_hops, depth=100):
        """
        执行一次完整的 N 跳搜索。

        Args:
            query (str): 初始的、用户提出的问题。
            num_hops (int): 要执行的跳数。
            depth (int, optional): 每一跳检索的候选段落数量。默认为 100。

        Returns:
            tuple:
                - final_facts: 最后一跳后，Condenser 输出的最终信息片段。
                - pids_bag: 在整个多跳过程中遇到的所有段落ID的集合。
                - stage1_preds: 最后一跳中，Condenser 阶段一的输出。
        """
        # 每一跳需要从 `depth` 个候选中精炼出 `k` 个新段落
        k = depth // num_hops

        # 初始化
        facts = [] # 存储 (pid, sid)
        context = None # 存储用于下一跳的上下文文本
        pids_bag = set() # 存储所有遇到过的 PID

        # --- 多跳循环 ---
        for hop_idx in range(num_hops):
            print(f"\n--- 第 {hop_idx + 1} 跳 ---")
            
            # 1. 检索 (Search)
            # 使用上一跳生成的 context 来增强当前查询
            ranking_results = self.searcher.search(query, context=context, k=depth)
            
            # 从检索结果中筛选出 `k` 个新的、之前未见过的段落用于冷凝
            pids_for_condenser = []
            facts_pids = {pid for pid, _ in facts}
            for pid, _, _ in zip(*ranking_results):
                if len(pids_for_condenser) < k and pid not in facts_pids:
                    pids_for_condenser.append(pid)
                if len(pids_bag) < k * (hop_idx + 1):
                    pids_bag.add(pid)
            
            # 2. 冷凝 (Condense)
            # 从这 `k` 个新段落中抽取出最关键的句子
            stage1_preds, facts, stage2_L3x = self.condenser.condense(query, backs=facts, ranking=pids_for_condenser)
            
            # 3. 准备下一跳的上下文
            # 将新找到的 facts 对应的句子文本拼接起来
            context = ' [SEP] '.join([self.collectionX.get((pid, sid), '') for pid, sid in facts])
            print(f"生成的下一跳上下文: {context[:200]}...")

        return stage2_L3x, pids_bag, stage1_preds