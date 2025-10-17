# 文件名: colbert/trainer.py

# 导入 ColBERT 的基础设施模块，用于管理实验运行、配置和分布式启动
from colbert.infra.run import Run
from colbert.infra.launcher import Launcher
from colbert.infra.config import ColBERTConfig

# 导入核心的训练逻辑函数
from colbert.training.training import train


class Trainer:
    """
    ColBERT 的主训练器类。

    这个类封装了启动模型训练所需的所有配置和数据。用户通过实例化这个类，
    配置相关参数，然后调用 `.train()` 方法来启动一个（可能是分布式的）训练任务。
    """

    def __init__(self, triples, queries, collection, config=None):
        """
        初始化 Trainer。

        Args:
            triples (str or Examples): 训练三元组文件的路径或 Examples 对象。
            queries (str or Queries): 查询文件的路径或 Queries 对象。
            collection (str or Collection): 文档集合的路径或 Collection 对象。
            config (ColBERTConfig, optional): 自定义配置对象。如果未提供，将使用默认配置。
        """
        # 从传入的 config 和全局的 Run().config 合并配置
        self.config = ColBERTConfig.from_existing(config, Run().config)

        # 保存数据源的引用
        self.triples = triples
        self.queries = queries
        self.collection = collection

    def configure(self, **kw_args):
        """
        使用给定的关键字参数更新训练配置。
        这允许在初始化后动态地修改训练参数。
        """
        self.config.configure(**kw_args)

    def train(self, checkpoint='bert-base-uncased'):
        """
        启动模型训练。

        Args:
            checkpoint (str, optional): 预训练模型的名称或本地检查点路径，作为训练的起点。
                                        默认为 'bert-base-uncased'。
                                        注意：这里传入的 checkpoint 会覆盖 config 中的设置。

        Returns:
            str: 训练完成后，验证性能最好的模型的检查点路径。
        """
        # 将数据源路径和初始检查点路径配置到 config 对象中
        self.configure(triples=self.triples, queries=self.queries, collection=self.collection)
        self.configure(checkpoint=checkpoint)

        # 使用 Launcher 来启动（可能是分布式的）训练任务
        # Launcher 会负责创建多个进程，并在每个进程中调用 `train` 函数
        launcher = Launcher(train)
        self._best_checkpoint_path = launcher.launch(self.config, self.triples, self.queries, self.collection)
        
        # 训练结束后，可以通过 .best_checkpoint_path() 获取最佳模型路径
        return self._best_checkpoint_path

    def best_checkpoint_path(self):
        """返回训练过程中验证性能最好的模型的检查点路径。"""
        return self._best_checkpoint_path