# 文件名: colbert/indexing/collection_encoder.py

import torch

from colbert.infra.run import Run
from colbert.utils.utils import batch


class CollectionEncoder:
    """
    负责将文档集合中的文本段落编码为 ColBERT 嵌入向量。
    
    这个类封装了与模型交互以生成嵌入向量的逻辑，并针对大规模处理进行了优化，
    例如通过分批处理来避免内存溢出。
    """

    def __init__(self, config, checkpoint):
        """
        初始化 CollectionEncoder。

        Args:
            config (ColBERTConfig): 配置对象。
            checkpoint (Checkpoint): 加载了预训练权重的 ColBERT 模型实例。
        """
        self.config = config
        self.checkpoint = checkpoint

    def encode_passages(self, passages):
        """
        将一批文本段落编码为嵌入向量。

        Args:
            passages (list[str]): 待编码的文本段落列表。

        Returns:
            tuple:
                - embs (torch.Tensor): 编码后的嵌入向量，形状为 (总词元数, dim)，以扁平化形式存储。
                - doclens (list[int]): 每个段落对应的词元数量列表。
        """
        Run().print(f"#> 正在编码 {len(passages)} 个段落...")

        if len(passages) == 0:
            Run().print(f"#> [RANK {self.config.rank}] 空段落列表，直接返回")
            return None, None

        with torch.inference_mode():  # 关闭梯度计算以加速并减少内存使用
            embs, doclens = [], []

            # 将段落分批送入模型，以避免因中间结果占用过多 GPU 显存而导致 OOM
            # 这里的批次大小是根据经验设置的，以在速度和内存之间取得平衡
            for passages_batch in batch(passages, self.config.bsize * 50):
                # 调用模型的 docFromText 方法进行编码
                # keep_dims='flatten' 表示返回一个扁平化的嵌入张量和每个文档的长度列表
                embs_, doclens_ = self.checkpoint.docFromText(passages_batch, bsize=self.config.bsize,
                                                              keep_dims='flatten', showprogress=True)
                embs.append(embs_)
                doclens.extend(doclens_)

            # 将所有批次的嵌入向量拼接成一个大的张量
            Run().print(f"#> [RANK {self.config.rank}] 编码完成，正在拼接 {len(embs)} 个张量...")
            embs = torch.cat(embs)
            Run().print(f"#> [RANK {self.config.rank}] 拼接完成！总嵌入: {embs.shape}")

        return embs, doclens