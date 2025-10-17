# 文件名: colbert/infra/utilities/minicorpus.py

import os
import random

# 导入工具和数据处理类
from colbert.utils.utils import create_directory
from colbert.data import Collection, Queries, Ranking


def sample_minicorpus(name, factor, topk=30, maxdev=3000):
    """
    一个用于从大规模数据集中采样创建一个“迷你”语料库的脚本。
    
    这对于快速进行原型开发、测试和调试非常有用，因为它创建了一个
    包含所有必要部分（训练集、开发集、文档集合）的小规模、自洽的数据集。

    Args:
        name (str): 迷你语料库的名称 (例如, 'nano', 'micro')。
        factor (int): 采样因子，用于控制采样的大小。
        topk (int, optional): 对于每个采样的查询，保留其排名前 k 的段落。
        maxdev (int, optional): 开发集查询的最大数量。
    """
    random.seed(12345)

    # 定义原始大规模数据集的路径
    # 注意：这些路径是硬编码的，需要根据实际情况修改
    COLLECTION_PATH = '/path/to/your/large/collection.tsv'
    QAS_TRAIN_PATH = '/path/to/your/large/train_qas.json'
    QAS_DEV_PATH = '/path/to/your/large/dev_qas.json'
    RANKING_TRAIN_PATH = '/path/to/your/large/train_ranking.tsv'
    RANKING_DEV_PATH = '/path/to/your/large/dev_ranking.tsv'

    # 加载原始数据
    collection = Collection(path=COLLECTION_PATH)
    qas_train = Queries(path=QAS_TRAIN_PATH).qas()
    qas_dev = Queries(path=QAS_DEV_PATH).qas()
    ranking_train = Ranking(path=RANKING_TRAIN_PATH).todict()
    ranking_dev = Ranking(path=RANKING_DEV_PATH).todict()

    # 从训练集和开发集中随机采样查询
    sample_train_qids = random.sample(list(qas_train.keys()), min(len(qas_train), 300 * factor))
    sample_dev_qids = random.sample(list(qas_dev.keys()), min(len(qas_dev), maxdev, 30 * factor))

    # 收集这些采样查询的 top-k 相关段落的 PID
    train_pids = {pid for qid in sample_train_qids for _, pid, *_ in ranking_train[qid][:topk]}
    dev_pids = {pid for qid in sample_dev_qids for _, pid, *_ in ranking_dev[qid][:topk]}
    
    sample_pids = sorted(list(train_pids.union(dev_pids)))
    print(f'# 采样的段落总数 = {len(sample_pids)}')

    # 设置新迷你语料库的输出根目录
    ROOT = f'/output/path/for/NQ-{name}'
    create_directory(os.path.join(ROOT, 'train'))
    create_directory(os.path.join(ROOT, 'dev'))

    # 保存新的查询集
    new_train_queries = Queries(data={qid: qas_train[qid] for qid in sample_train_qids})
    new_train_queries.save_qas(os.path.join(ROOT, 'train/qas.json'))

    new_dev_queries = Queries(data={qid: qas_dev[qid] for qid in sample_dev_qids})
    new_dev_queries.save_qas(os.path.join(ROOT, 'dev/qas.json'))

    # 保存新的文档集合
    print(f"正在将新的集合保存至 {os.path.join(ROOT, 'collection.tsv')}")
    new_collection_data = [collection[pid] for pid in sample_pids]
    Collection(data=new_collection_data).save(os.path.join(ROOT, 'collection.tsv'))

    print('#> 完成！')