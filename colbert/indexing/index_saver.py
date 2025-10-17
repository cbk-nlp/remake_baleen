# 文件名: colbert/indexing/index_saver.py

import os
import queue
import ujson
import threading
from contextlib import contextmanager

from colbert.indexing.codecs.residual import ResidualCodec


class IndexSaver:
    """
    负责将索引块异步写入磁盘的类。

    在索引构建过程中，文档编码（在 GPU 上）和文件写入（I/O 操作）是两个主要步骤。
    为了避免 I/O 操作成为瓶颈，这个类使用一个单独的线程来处理文件保存任务。
    主线程可以将编码好的数据块放入一个队列，然后继续进行下一个数据块的编码，
    而保存线程则从队列中取出数据并写入文件。
    """

    def __init__(self, config):
        """
        初始化 IndexSaver。

        Args:
            config (ColBERTConfig): 配置对象。
        """
        self.config = config

    def save_codec(self, codec):
        """保存残差编解码器 (ResidualCodec)。"""
        codec.save(index_path=self.config.index_path_)

    def load_codec(self):
        """加载残差编解码器 (ResidualCodec)。"""
        return ResidualCodec.load(index_path=self.config.index_path_)

    @contextmanager
    def thread(self):
        """
        一个上下文管理器，用于启动和管理保存线程。

        用法:
        with saver.thread():
            # 在这里调用 save_chunk
            ...
        """
        self.codec = self.load_codec()
        self.saver_queue = queue.Queue(maxsize=3)  # 队列大小设为3，作为缓冲区
        
        # 创建并启动保存线程
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        try:
            yield
        finally:
            # 发送一个 None 信号，通知线程结束
            self.saver_queue.put(None)
            thread.join()  # 等待线程执行完毕
            del self.saver_queue
            del self.codec

    def save_chunk(self, chunk_idx, offset, embs, doclens):
        """
        压缩一个嵌入块并将其放入保存队列。

        Args:
            chunk_idx (int): 索引块的编号。
            offset (int): 该块在整个集合中的起始文档偏移量。
            embs (torch.Tensor): 待压缩的嵌入向量。
            doclens (list[int]): 每个文档的长度列表。
        """
        # 在放入队列前进行压缩，因为压缩是 CPU 密集型操作
        compressed_embs = self.codec.compress(embs)
        self.saver_queue.put((chunk_idx, offset, compressed_embs, doclens))

    def _saver_thread(self):
        """保存线程的主循环。它会不断从队列中获取数据并写入磁盘，直到收到 None。"""
        for args in iter(self.saver_queue.get, None):
            self._write_chunk_to_disk(*args)

    def _write_chunk_to_disk(self, chunk_idx, offset, compressed_embs, doclens):
        """
        将单个压缩后的索引块及其元数据写入磁盘。

        每个块会生成三个文件：
        1.  `{chunk_idx}.codes.pt` 和 `{chunk_idx}.residuals.pt`: 压缩后的嵌入数据。
        2.  `doclens.{chunk_idx}.json`: 该块中文档的长度信息。
        3.  `{chunk_idx}.metadata.json`: 该块的元数据，如偏移量、文档数等。
        """
        path_prefix = os.path.join(self.config.index_path_, str(chunk_idx))
        compressed_embs.save(path_prefix)

        doclens_path = os.path.join(self.config.index_path_, f'doclens.{chunk_idx}.json')
        with open(doclens_path, 'w') as f:
            ujson.dump(doclens, f)

        metadata_path = os.path.join(self.config.index_path_, f'{chunk_idx}.metadata.json')
        with open(metadata_path, 'w') as f:
            metadata = {'passage_offset': offset, 'num_passages': len(doclens),
                        'num_embeddings': len(compressed_embs)}
            ujson.dump(metadata, f)