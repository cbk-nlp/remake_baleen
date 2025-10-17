# 文件名: colbert/utils/parser.py

import os
import copy
import faiss
from argparse import ArgumentParser

import colbert.utils.distributed as distributed
from colbert.utils.runs import Run
from colbert.utils.utils import print_message


class Arguments:
    """
    一个基于 Python 内置 `ArgumentParser` 的包装类。

    它为 ColBERT 项目提供了一个结构化的方式来定义和解析命令行参数。
    通过将相关的参数组合到不同的添加方法中（如 `add_model_parameters`），
    使得为不同的脚本（训练、索引、搜索）配置参数变得更加模块化和清晰。
    """

    def __init__(self, description):
        self.parser = ArgumentParser(description=description)
        self.checks = [] # 用于存储参数验证函数

        # --- 添加通用的实验管理参数 ---
        self.add_argument('--root', dest='root', default='experiments', help="实验的根目录")
        self.add_argument('--experiment', dest='experiment', default='default', help="实验的名称")
        self.add_argument('--run', dest='run', default=Run.name, help="本次运行的唯一名称（通常是时间戳）")
        self.add_argument('--local_rank', dest='rank', default=-1, type=int, help="PyTorch 分布式训练的 rank")

    def add_model_parameters(self):
        """添加与 ColBERT 模型架构相关的核心参数。"""
        self.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'], help="相似度计算方法")
        self.add_argument('--dim', dest='dim', default=128, type=int, help="ColBERT 嵌入的维度")
        self.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int, help="查询的最大长度")
        self.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int, help="文档的最大长度")
        self.add_argument('--mask-punctuation', dest='mask_punctuation', action='store_true', help="是否在编码时掩码标点符号")

    def add_model_training_parameters(self):
        """添加与模型训练过程相关的参数。"""
        self.add_argument('--resume', dest='resume', action='store_true', help="从最新的检查点恢复训练")
        self.add_argument('--checkpoint', dest='checkpoint', default=None, help="作为训练起点的模型检查点路径")
        self.add_argument('--lr', dest='lr', default=3e-06, type=float, help="学习率")
        self.add_argument('--maxsteps', dest='maxsteps', default=400000, type=int, help="最大训练步数")
        self.add_argument('--bsize', dest='bsize', default=32, type=int, help="批次大小")
        self.add_argument('--accum', dest='accumsteps', default=1, type=int, help="梯度累积步数")
        self.add_argument('--amp', dest='amp', action='store_true', help="启用自动混合精度训练")

    # ... (其他 add_* 方法与此类似，用于添加不同场景下的参数) ...

    def add_argument(self, *args, **kw_args):
        """直接调用底层的 ArgumentParser.add_argument 方法。"""
        return self.parser.add_argument(*args, **kw_args)

    def check_arguments(self, args):
        """执行所有注册的参数验证函数。"""
        for check in self.checks:
            check(args)

    def parse(self):
        """
        解析所有定义的命令行参数，并进行后处理。

        后处理包括：
        1.  验证参数。
        2.  初始化分布式环境。
        3.  设置 FAISS 的线程数。
        4.  初始化全局的 Run 对象并记录参数。
        """
        args = self.parser.parse_args()
        self.check_arguments(args)

        # 创建一个输入参数的深拷贝，用于日志记录
        args.input_arguments = copy.deepcopy(args)

        # 初始化分布式环境
        args.nranks, args.distributed = distributed.init(args.rank)

        # 为 FAISS 设置合适的线程数，以避免在多进程环境中超载 CPU
        if args.nranks > 1:
            num_threads = max(1, int(os.cpu_count() / args.nranks))
            print_message(f"#> 在分布式环境中，将每个进程的 FAISS 线程数限制为 {num_threads}", condition=(args.rank == 0))
            faiss.omp_set_num_threads(num_threads)

        # 初始化全局 Run 对象
        Run.init(args.rank, args.root, args.experiment, args.run)
        # 记录参数
        Run._log_args(args)
        Run.info("输入的参数:", args.input_arguments.__dict__)

        return args