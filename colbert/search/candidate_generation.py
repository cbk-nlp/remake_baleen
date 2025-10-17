# 文件名: colbert/search/candidate_generation.py

import torch

from colbert.search.strided_tensor import StridedTensor
from .strided_tensor_core import _create_mask, _create_view


class CandidateGeneration:
    """
    负责 ColBERT 检索流程的第一阶段：候选生成。

    这一阶段利用了倒排文件（IVF）索引来快速缩小搜索范围。步骤如下：
    1.  **Probing**: 对于一个查询，将其与所有聚类中心进行比较，找到最相近的 `nprobe` 个聚类中心。
    2.  **Candidate Gathering**: 从这 `nprobe` 个聚类中心对应的倒排列表中，收集所有词元嵌入的 ID (eid)。
    3.  **Candidate Scoring**: 计算查询嵌入与所有收集到的候选词元嵌入之间的相似度。
    4.  **Aggregation**: 按文档（PID）聚合词元级别的分数，为每个候选文档计算一个初步分数。
    5.  **Pruning**: 根据初步分数，选取 top-`ncandidates` 个文档作为最终的候选集，送入下一阶段进行精确重排。
    """

    def generate_candidate_eids(self, Q, nprobe):
        """
        步骤 1 & 2: Probing 和 Candidate Gathering。
        找到与查询最相关的聚类中心，并收集这些中心对应的所有嵌入 ID (eids)。
        """
        # 计算查询与所有聚类中心的相似度，并取 top-k (nprobe)
        cells = (self.codec.centroids @ Q.T).topk(nprobe, dim=0, sorted=False).indices.permute(1, 0)
        cells = cells.flatten().contiguous().unique(sorted=False)

        # 从倒排文件 (IVF) 中查找这些 cell 对应的嵌入 ID 列表
        eids, cell_lengths = self.ivf.lookup(cells)
        return eids.cuda(), cells.cuda(), cell_lengths.cuda()

    def generate_candidate_scores(self, nprobe, Q, eids, cells, cell_lengths):
        """
        步骤 3: Candidate Scoring。
        计算查询嵌入与候选嵌入之间的相似度分数。
        """
        eids = torch.unique(eids.cuda().long(), sorted=False)
        # 从索引中解压缩这些候选嵌入
        E = self.lookup_eids(eids).cuda()

        # 计算查询中每个词元与所有候选词元之间的点积相似度
        scores = (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T
        return scores.cuda(), eids

    def generate_candidates(self, config, Q):
        """
        执行完整的候选生成流程 (步骤 1-5)。
        """
        nprobe = config.nprobe
        ncandidates = config.ncandidates
        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0).cuda().half()
        
        # 步骤 1 & 2
        eids, cells, cell_lengths = self.generate_candidate_eids(Q, nprobe)
        # 步骤 3
        scores, eids = self.generate_candidate_scores(nprobe, Q, eids, cells, cell_lengths)
        
        # 将词元 ID (eid) 映射到文档 ID (pid)
        pids = self.emb2pid[eids.long()].cuda()

        # 步骤 4: Aggregation (按 PID 聚合分数)
        # 这是一个高度优化的实现，它通过排序和 `unique_consecutive` 来高效地对分数进行分组
        sorter = pids.sort()
        pids, scores = sorter.values, torch.take_along_dim(scores, sorter.indices.unsqueeze(0), dim=-1)
        pids_unique, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        
        if len(pids_unique) <= ncandidates:
            return pids_unique

        # 使用 StridedTensor 的技巧来高效地对每个 PID 对应的分数求和
        pids_offsets = pids_counts.cumsum(dim=0) - pids_counts[0]
        stride = pids_counts.max().item()
        scores = torch.nn.functional.pad(scores, (0, stride)).cuda()
        
        q_scores = []
        for idx in range(scores.size(0)):
            scores_padded = _create_view(scores[idx], stride, [])[pids_offsets] * _create_mask(pids_counts, stride)
            # 对每个文档，取其所有词元与查询的最大相似度得分，然后求和
            scores_maxsim = scores_padded.max(-1).values
            q_scores.append(scores_maxsim)
        
        # ... (此处省略了更复杂的聚合逻辑，核心思想是得到每个候选 PID 的总分) ...
        # 最终得到每个候选 PID 的聚合分数 `scores_lb`
        scores_lb = torch.stack(q_scores).sum(0)
        assert scores_lb.size(0) == pids_unique.size(0)

        # 步骤 5: Pruning
        # 如果候选数量大于 ncandidates，则取分数最高的 ncandidates 个
        if scores_lb.size(0) > ncandidates:
            return pids_unique[scores_lb.topk(ncandidates, dim=-1).indices]
        
        return pids_unique