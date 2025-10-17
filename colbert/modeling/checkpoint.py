# 文件名: colbert/modeling/checkpoint.py

import torch
from tqdm import tqdm

from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager
from colbert.modeling.colbert import ColBERT, colbert_score


class Checkpoint(ColBERT):
    """
    一个专门用于推理（Inference）的 ColBERT 模型封装类。

    它继承自 ColBERT，并提供了更便捷的 API 来直接从文本编码查询和文档，
    而无需手动处理分词和张量化。此类在 `Indexer` 和 `Searcher` 中被广泛使用。
    """

    def __init__(self, name, colbert_config=None):
        """
        初始化 Checkpoint。

        Args:
            name (str): 预训练模型的名称或本地模型路径。
            colbert_config (ColBERTConfig, optional): 自定义配置对象。
        """
        super().__init__(name, colbert_config)
        # 确认模型处于评估模式
        assert not self.training, "Checkpoint 实例必须处于评估模式"

        # 初始化专门用于查询和文档的分词器
        self.query_tokenizer = QueryTokenizer(self.colbert_config)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)

        # 初始化自动混合精度（AMP）管理器，以加速推理
        self.amp_manager = MixedPrecisionManager(True)

    def query(self, *args, to_cpu=False, **kw_args):
        """
        编码一个或多个查询。这是一个底层的编码接口。

        Args:
            to_cpu (bool, optional): 是否将结果张量移动到 CPU。默认为 False。
        
        Returns:
            torch.Tensor: 查询的嵌入矩阵。
        """
        with torch.no_grad(): # 推理时不需要计算梯度
            with self.amp_manager.context(): # 使用混合精度
                Q = super().query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        """
        编码一个或多个文档。这是一个底层的编码接口。

        Args:
            to_cpu (bool, optional): 是否将结果张量移动到 CPU。默认为 False。
        
        Returns:
            torch.Tensor or tuple: 文档的嵌入矩阵（可能还包含其他信息，如掩码）。
        """
        with torch.no_grad():
            with self.amp_manager.context():
                D = super().doc(*args, **kw_args)
                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()
                return D

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None):
        """
        从原始文本字符串列表编码查询。这是一个高级 API。

        Args:
            queries (list[str]): 待编码的查询字符串列表。
            bsize (int, optional): 批处理大小。如果提供，将分批处理以节省内存。
            to_cpu (bool, optional): 是否将最终结果移动到 CPU。
            context (list[str], optional): 与每个查询相关联的上下文文本。

        Returns:
            torch.Tensor: 所有查询的嵌入矩阵拼接成的张量。
        """
        if bsize:
            # 分批处理
            batches = self.query_tokenizer.tensorize(queries, context=context, bsize=bsize)
            batches_Q = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches_Q)

        # 不分批，一次处理
        input_ids, attention_mask = self.query_tokenizer.tensorize(queries, context=context)
        return self.query(input_ids, attention_mask)

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        """
        从原始文本字符串列表编码文档。这是一个高级 API。

        Args:
            docs (list[str]): 待编码的文档字符串列表。
            bsize (int, optional): 批处理大小。
            keep_dims (bool or str): 如何处理输出维度。
                                     True: 返回填充到同样长度的3D张量。
                                     False: 返回一个 Python 列表，每个元素是变长的2D张量。
                                     'flatten': 返回一个扁平化的2D嵌入张量和每个文档的长度列表。
            to_cpu (bool, optional): 是否将最终结果移动到 CPU。
            showprogress (bool, optional): 是否显示进度条。
            return_tokens (bool, optional): 是否同时返回编码后的 token ID。

        Returns:
            根据 `keep_dims` 的设置，返回不同格式的嵌入表示。
        """
        assert keep_dims in [True, False, 'flatten']

        if bsize:
            # 分词并分批
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)
            
            # 处理 `return_tokens` 选项
            returned_text = []
            if return_tokens:
                # ... (此处省略了 token ID 返回的详细逻辑) ...
                pass

            # 根据 keep_dims 设置不同的编码模式
            keep_dims_ = 'return_mask' if keep_dims == 'flatten' else keep_dims
            batches_D = [self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
                         for input_ids, attention_mask in tqdm(text_batches, disable=not showprogress)]

            # 根据 keep_dims 对结果进行后处理
            if keep_dims is True:
                D = _stack_3D_tensors(batches_D)
                return (D[reverse_indices], *returned_text)
            elif keep_dims == 'flatten':
                D_packed, mask_packed = zip(*batches_D)
                D, mask = torch.cat(D_packed)[reverse_indices], torch.cat(mask_packed)[reverse_indices]
                doclens = mask.squeeze(-1).sum(-1).tolist()
                D_flat = D[mask.bool().flatten()].cpu()
                return (D_flat, doclens, *returned_text)
            else: # keep_dims is False
                D_list = [d for batch in batches_D for d in batch]
                return ([D_list[idx] for idx in reverse_indices.tolist()], *returned_text)

        # 不分批处理
        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    def score(self, Q, D, mask=None, lengths=None):
        """
        计算查询和文档之间的 ColBERT 相关性分数 (MaxSim)。
        """
        # 确保 mask 和 lengths 不会同时提供
        if lengths is not None:
            assert mask is None, "不能同时提供 mask 和 lengths"
            # 根据 lengths 创建掩码
            mask = torch.arange(D.size(1), device=self.device).unsqueeze(0) + 1
            mask = mask <= lengths.to(self.device).unsqueeze(-1)

        # 直接调用核心的 colbert_score 函数
        return colbert_score(Q, D, mask, config=self.colbert_config)


def _stack_3D_tensors(groups):
    """一个辅助函数，用于将多个不同长度的3D张量堆叠并填充成一个大的3D张量。"""
    bsize = sum(x.size(0) for x in groups)
    maxlen = max(x.size(1) for x in groups)
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output