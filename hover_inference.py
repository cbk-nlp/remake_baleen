import os
import tqdm
import json
import random
import argparse

from colbert.data import Queries
from colbert.infra import Run, RunConfig

from baleen.condenser.condense import Condenser
from baleen.hop_searcher import HopSearcher
from baleen.engine import Baleen

from colbert.utils.utils import print_message


def main(args):
    """
    主函数，用于执行推理过程。
    """
    print_message("#> Starting...")

    # 定义数据和模型路径
    collectionX_path = os.path.join(args.datadir, 'wiki.abstracts.2017/collection.json')
    queries_path = os.path.join(args.datadir, 'hover/dev/questions.tsv')
    qas_path = os.path.join(args.datadir, 'hover/dev/qas.json')

    checkpointL1 = os.path.join(args.datadir, 'hover.checkpoints-v1.0/condenserL1-v1.0.dnn')
    checkpointL2 = os.path.join(args.datadir, 'hover.checkpoints-v1.0/condenserL2-v1.0.dnn')

    # 在运行上下文中初始化搜索器、冷凝器和Baleen引擎
    with Run().context(RunConfig(root=args.root)):
        searcher = HopSearcher(index=args.index)
        condenser = Condenser(checkpointL1=checkpointL1, checkpointL2=checkpointL2,
                              collectionX_path=collectionX_path, deviceL1='cuda:0', deviceL2='cuda:0')

        baleen = Baleen(collectionX_path, searcher, condenser)
        baleen.searcher.configure(nprobe=2, ncandidates=8192)

    # 加载查询
    queries = Queries(path=queries_path)
    outputs = {}

    # 遍历查询并执行搜索
    for qid, query in tqdm.tqdm(queries.items()):
        facts, pids_bag, _ = baleen.search(query, num_hops=4)
        outputs[qid] = (facts, pids_bag)

    # 将输出写入JSON文件
    with Run().open('output.json', 'w') as f:
        f.write(json.dumps(outputs) + '\n')


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="根目录路径")
    parser.add_argument("--datadir", type=str, required=True, help="数据目录路径")
    parser.add_argument("--index", type=str, required=True, help="索引名称")

    args = parser.parse_args()
    main(args)