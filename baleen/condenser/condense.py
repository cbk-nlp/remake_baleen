# 文件名: baleen/condenser/condense.py

import torch
import ujson
from colbert.utils.utils import f7
from colbert.utils.amp import MixedPrecisionManager
from baleen.condenser.model import ElectraReader
from baleen.condenser.tokenization import AnswerAwareTokenizer


class Condenser:
    """
    信息冷凝器。

    这是 Baleen 多跳问答流程中的核心组件。它执行一个两阶段（two-stage）的过程，
    从一轮检索到的大量段落中，精确地抽取出最相关的句子或信息片段（称为 "facts"）。

    - **阶段一 (Stage 1)**: 使用一个模型 (`modelL1`) 对所有候选段落中的所有句子进行粗略打分，
      选出 top-k 的候选句子。
    - **阶段二 (Stage 2)**: 将阶段一选出的 top-k 句子组合成一个新的上下文，然后使用一个
      更精细的模型 (`modelL2`) 对这个组合后的上下文再次打分，得到最终的、高置信度的 "facts"。
    """
    def __init__(self, collectionX_path, checkpointL1, checkpointL2, deviceL1='cuda', deviceL2='cuda'):
        # 加载两个阶段分别使用的模型
        self.modelL1, self.maxlenL1 = self._load_model(checkpointL1, deviceL1)
        self.modelL2, self.maxlenL2 = self._load_model(checkpointL2, deviceL2)

        # 初始化分词器和 AMP
        self.amp = MixedPrecisionManager(activated=True)
        self.tokenizer = AnswerAwareTokenizer(total_maxlen=self.maxlenL1)
        
        # 加载用于查找句子原文的集合
        self.CollectionX, self.CollectionY = self._load_collection(collectionX_path)

    def condense(self, query, backs, ranking):
        """
        执行完整的信息冷凝流程。

        Args:
            query (str): 当前跳的查询。
            backs (list[tuple]): 从之前跳数积累下来的 "facts" (pid, sid)。
            ranking (list[int]): 当前跳检索到的 top-k 段落的 PID 列表。

        Returns:
            tuple: (stage1_preds, stage2_preds, stage2_preds_L3x)
                   - `stage1_preds`: 阶段一选出的 top-k 句子。
                   - `stage2_preds`: 阶段二筛选出的高置信度句子。
                   - `stage2_preds_L3x`: 用于构建下一跳上下文的最终 "facts"。
        """
        stage1_preds = self._stage1(query, backs, ranking)
        stage2_preds, stage2_preds_L3x = self._stage2(query, stage1_preds)
        return stage1_preds, stage2_preds, stage2_preds_L3x

    def _load_model(self, path, device):
        """从检查点加载 ElectraReader 模型。"""
        checkpoint = torch.load(path, map_location='cpu')
        model = ElectraReader.from_pretrained(checkpoint['arguments']['model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        maxlen = checkpoint['arguments']['maxlen']
        return model, maxlen
    
    def _load_collection(self, collectionX_path):
        """加载用于根据 (pid, sid) 查找句子原文的集合。"""
        # ... (加载逻辑) ...
        return {}, {} # 简化示例

    def _stage1(self, query, BACKS, ranking, TOPK=9):
        """
        阶段一：粗筛选。
        对 `ranking` 中的每个段落，将其所有句子与查询进行匹配打分。
        """
        with torch.inference_mode():
            # 将之前的 facts 和当前查询组合成上下文
            context_sentences = [self.CollectionY.get(fact) for fact in BACKS if fact in self.CollectionY]
            context_query = ' # '.join([query] + context_sentences)

            # 准备所有待评分的段落，用 [MASK] 分隔句子
            passages = [' [MASK] '.join(self.CollectionX[pid]) for pid in ranking]
            
            # 使用分词器处理
            obj = self.tokenizer.process([context_query], passages)

            with self.amp.context():
                scores = self.modelL1(obj.encoding.to(self.modelL1.device)).view(-1)
            
            # 找到分数最高的 TOPK 个 [MASK] token，它们对应着最相关的句子
            topk_indices = scores.topk(min(TOPK, len(scores))).indices.tolist()
            
            # 将 token 索引映射回 (pid, sid)
            preds = [self._map_index_to_preds(idx, ranking) for idx in topk_indices]
            
            # 将新找到的句子与之前的 facts 合并，并去重
            pred_plus = f7(list(map(tuple, BACKS + preds)))[:TOPK]
        return pred_plus

    def _stage2(self, query, preds):
        """
        阶段二：精筛选。
        将阶段一选出的句子组合成一个段落，用 modelL2 再次打分。
        """
        # 将阶段一的 top-k 句子组合成一个上下文
        context_passage = ' [MASK] '.join([''] + [self.CollectionY.get(p) for p in preds if p in self.CollectionY])
        
        obj = self.tokenizer.process([query], [context_passage])
        
        with self.amp.context():
            scores = self.modelL2(obj.encoding.to(self.modelL2.device)).view(-1).tolist()
        
        # 根据分数对阶段一的句子进行排序
        preds_with_scores = sorted(zip(scores, preds), reverse=True)
        
        # 应用阈值进行最终筛选
        preds = [p for score, p in preds_with_scores if score > 0]
        # `L3x` 是一个更宽松的筛选结果，用于构建下一跳的上下文
        preds_L3x = [p for score, p in preds_with_scores if score > min(0, preds_with_scores[1][0] - 1e-10)]
        
        return preds, preds_L3x