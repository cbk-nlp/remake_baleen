import os
import argparse

from colbert.infra import Run, ColBERTConfig, RunConfig
from colbert import Indexer

from colbert.utils.utils import print_message


def main(args):
    """
    主函数，用于执行索引过程。
    """
    print_message("#> Starting...")

    # 定义数据和模型路径
    collection_path = os.path.join(args.datadir, 'wiki.abstracts.2017/collection.tsv')
    checkpoint_path = os.path.join(args.datadir, 'hover.checkpoints-v1.0/flipr-v1.0.dnn')

    index_path = os.path.join(args.datadir, args.index)
    # 在运行上下文中初始化索引器并创建索引
    with Run().context(RunConfig(root=args.root, nranks=6, overwrite=True)):
        config = ColBERTConfig(doc_maxlen=256, nbits=args.nbits, rank=0, nranks=6)
        indexer = Indexer(checkpoint_path, config=config)
        indexer.index(name=args.index, index=index_path, collection=collection_path, overwrite=True)


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="根目录路径")
    parser.add_argument("--datadir", type=str, required=True, help="数据目录路径")
    parser.add_argument("--index", type=str, required=True, help="索引名称")
    parser.add_argument("--nbits", type=int, required=True, help="用于压缩的位数")

    args = parser.parse_args()
    main(args)

'''
python hover_indexing.py --root /home/bkcai/task/hover/remake_baleen --datadir /data/bkcai/data/hover --index wiki17.hover.2bit --nbits 2
'''