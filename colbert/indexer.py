# 文件名: colbert/indexer.py

import os
import time
import torch.multiprocessing as mp

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig
from colbert.infra.launcher import Launcher
from colbert.utils.utils import create_directory, print_message
from colbert.indexing.collection_indexer import encode


class Indexer:
    """
    ColBERT 的主索引器类。

    该类封装了从文档集合创建 ColBERT 索引所需的所有步骤。
    它管理配置、启动分布式索引进程，并返回最终的索引路径。
    用户通过实例化这个类并调用 .index() 方法来构建索引。
    """

    def __init__(self, checkpoint, config=None):
        """
        初始化 Indexer。

        Args:
            checkpoint (str): 用于文档编码的预训练 ColBERT 模型的路径。
            config (ColBERTConfig, optional): 一个 ColBERTConfig 对象，用于自定义配置。
                                              如果未提供，将使用默认配置。
        """
        self.index_path = None
        self.checkpoint = checkpoint

        # 从检查点加载配置，并与传入的 config 和全局 Run().config 合并
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, Run().config, config)
        
        # 应用检查点配置
        self.configure(checkpoint=checkpoint)

    def configure(self, **kw_args):
        """用给定的关键字参数更新配置。"""
        self.config.configure(**kw_args)

    def get_index(self):
        """返回构建完成的索引的路径。"""
        return self.index_path

    def erase(self):
        """
        删除索引目录中已存在的索引文件。
        
        为了防止意外删除，该操作会等待20秒让用户确认。
        它主要删除元数据（.json）和索引数据（.pt）文件。
        """
        assert self.index_path is not None, "索引路径尚未设置"
        directory = self.index_path
        deleted = []

        for filename in sorted(os.listdir(directory)):
            filename = os.path.join(directory, filename)

            # 定义要删除的文件类型
            is_metadata = filename.endswith(".json") and ('metadata' in filename or 'doclen' in filename or 'plan' in filename)
            is_data = filename.endswith(".pt")
            
            if is_metadata or is_data:
                deleted.append(filename)
        
        if len(deleted):
            print_message(f"#> 将在 20 秒后删除位于 {directory} 的 {len(deleted)} 个文件...")
            time.sleep(20)

            for filename in deleted:
                os.remove(filename)

        return deleted

    def index(self, name, index, collection, overwrite=False):
        """
        为给定的文档集合构建索引。

        Args:
            name (str): 索引的名称。将作为索引目录的子目录名。
            collection (Collection or str): 文档集合对象或其路径。
            overwrite (bool, optional): 如果为 True，将删除已存在的同名索引。默认为 False。

        Returns:
            str: 构建完成的索引的路径。
        """
        # 配置索引名称和集合路径
        self.configure(collection=collection, index_name=name, index_path=index)
        # 索引构建过程中的默认参数
        self.configure(bsize=256, partitions=None)

        self.index_path = self.config.index_path_

        # 检查索引路径是否存在
        if not overwrite:
            assert not os.path.exists(self.config.index_path_), f"索引路径 {self.config.index_path_} 已存在！请使用 overwrite=True 或选择其他名称。"
        
        create_directory(self.config.index_path_)

        if overwrite:
            self.erase()

        # 启动分布式索引构建过程
        self.__launch(collection)

        return self.index_path

    def __launch(self, collection):
        """
        使用 Launcher 启动多进程（分布式）的索引编码任务。
        """
        # 使用多进程管理器创建共享对象，用于进程间通信
        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(self.config.nranks)]
        shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]

        # Launcher 负责在多个 rank (GPU) 上并行执行 `encode` 函数
        launcher = Launcher(encode)
        launcher.launch(self.config, collection, shared_lists, shared_queues)