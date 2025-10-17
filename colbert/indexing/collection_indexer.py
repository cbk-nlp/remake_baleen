# 文件名: colbert/indexing/collection_indexer.py

import os
import tqdm
import ujson
import faiss
import torch
import random
import numpy as np
import torch.multiprocessing as mp

from colbert.infra.config.config import ColBERTConfig
import colbert.utils.distributed as distributed
from colbert.infra.run import Run
from colbert.infra.launcher import print_memory_stats
from colbert.modeling.checkpoint import Checkpoint
from colbert.data.collection import Collection
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.indexing.index_saver import IndexSaver
from colbert.utils.utils import print_message
from colbert.indexing.codecs.residual import ResidualCodec


def encode(config, collection, shared_lists, shared_queues):
    """
    一个独立的顶层函数，作为多进程任务的入口点。
    它实例化并运行 CollectionIndexer。
    """
    encoder = CollectionIndexer(config=config, collection=collection)
    encoder.run(shared_lists)


class CollectionIndexer:
    """
    负责协调整个文档集合索引构建过程的类。

    这个过程分为几个主要阶段：
    1.  **Setup**: 设置分布式环境，对一小部分文档进行采样以估计平均文档长度，并规划索引的分区数。
    2.  **Train**: 使用采样得到的嵌入向量训练 K-means 模型，以生成用于量化的聚类中心。
                   同时计算残差量化的参数（桶边界和权重）。
    3.  **Index**: 遍历整个文档集合，将每个文档编码为嵌入向量，然后压缩并保存到磁盘上的索引块中。
    4.  **Finalize**: 合并所有索引块的信息，构建倒排文件（IVF），并写入最终的元数据。
    """

    def __init__(self, config: ColBERTConfig, collection):
        self.config = config
        self.rank, self.nranks = self.config.rank, self.config.nranks

        if self.config.rank == 0:
            self.config.help()

        self.collection = Collection.cast(collection)
        self.checkpoint = Checkpoint(self.config.checkpoint, colbert_config=self.config).cuda()
        self.encoder = CollectionEncoder(config, self.checkpoint)
        self.saver = IndexSaver(config)

        print_memory_stats(f'RANK:{self.rank}')

    def run(self, shared_lists):
        """执行完整的索引构建流程。"""
        with torch.inference_mode():
            # 依次执行各个阶段，并在阶段之间设置屏障（barrier）以同步所有进程
            self.setup()
            distributed.barrier(self.rank)
            
            self.train(shared_lists)
            distributed.barrier(self.rank)
            
            self.index()
            distributed.barrier(self.rank)
            
            self.finalize()
            distributed.barrier(self.rank)

    def setup(self):
        """准备阶段：采样文档，估计参数，并保存索引计划。"""
        self.num_chunks = int(np.ceil(len(self.collection) / self.collection.get_chunksize()))
        sampled_pids = self._sample_pids()
        avg_doclen_est = self._sample_embeddings(sampled_pids)

        # 根据集合大小和平均文档长度估算总嵌入数，并以此决定 IVF 的分区数
        num_passages = len(self.collection)
        self.num_embeddings_est = num_passages * avg_doclen_est
        # 经验公式，分区数通常与总嵌入数的平方根成正比
        self.num_partitions = int(2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est))))

        Run().print_main(f'创建 {self.num_partitions:,} 个分区。')
        Run().print_main(f'*估计* 有 {int(self.num_embeddings_est):,} 个嵌入。')

        self._save_plan()

    def _sample_pids(self):
        """从整个集合中随机采样一小部分 PID 用于训练 K-means。"""
        num_passages = len(self.collection)
        # 采样数量的经验公式
        sampled_pids_count = min(1 + int(16 * np.sqrt(120 * num_passages)), num_passages)
        sampled_pids = random.sample(range(num_passages), sampled_pids_count)
        Run().print_main(f"# 采样 PID 数量 = {len(sampled_pids)}")
        return set(sampled_pids)

    def _sample_embeddings(self, sampled_pids):
        """对采样的 PID 对应的文档进行编码，以获得用于训练的嵌入向量。"""
        # 每个进程只处理分配给自己的采样文档
        local_pids = self.collection.enumerate(rank=self.rank)
        local_sample = [passage for pid, passage in local_pids if pid in sampled_pids]
        local_sample_embs, doclens = self.encoder.encode_passages(local_sample)

        # 使用 all_reduce 聚合所有进程的信息
        self.num_sample_embs = torch.tensor([local_sample_embs.size(0)]).cuda()
        torch.distributed.all_reduce(self.num_sample_embs)
        
        avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
        avg_doclen_est = torch.tensor([avg_doclen_est]).cuda()
        torch.distributed.all_reduce(avg_doclen_est)

        nonzero_ranks = torch.tensor([float(len(local_sample) > 0)]).cuda()
        torch.distributed.all_reduce(nonzero_ranks)

        self.avg_doclen_est = avg_doclen_est.item() / nonzero_ranks.item()
        Run().print(f'估计的平均文档长度 = {self.avg_doclen_est:.2f}')

        # 将本地采样的嵌入保存到临时文件
        torch.save(local_sample_embs, os.path.join(self.config.index_path_, f'sample.{self.rank}.pt'))
        return self.avg_doclen_est

    def _save_plan(self):
        """由主进程 (rank 0) 保存索引计划文件 (plan.json)，记录关键参数。"""
        if self.rank < 1:
            plan_path = os.path.join(self.config.index_path_, 'plan.json')
            Run().print("#> 正在将索引计划保存至", plan_path, "..")
            with open(plan_path, 'w') as f:
                d = {'config': self.config.export(), 'num_chunks': self.num_chunks,
                     'num_partitions': self.num_partitions, 'num_embeddings_est': self.num_embeddings_est,
                     'avg_doclen_est': self.avg_doclen_est}
                f.write(ujson.dumps(d, indent=4) + '\n')

    def train(self, shared_lists):
        """训练阶段：仅由主进程执行 K-means 训练和残差参数计算。"""
        if self.rank > 0:
            return

        sample, heldout = self._concatenate_and_split_sample()
        centroids = self._train_kmeans(sample, shared_lists)
        del sample

        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(centroids, heldout)
        print_message(f'平均残差 = {avg_residual}')
        
        codec = ResidualCodec(config=self.config, centroids=centroids, avg_residual=avg_residual,
                              bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights)
        self.saver.save_codec(codec)

    def _concatenate_and_split_sample(self):
        """合并所有进程保存的采样嵌入，并划分为训练集和验证集（heldout）。"""
        sample = torch.empty(self.num_sample_embs, self.config.dim, dtype=torch.float16)
        offset = 0
        for r in range(self.nranks):
            sub_sample_path = os.path.join(self.config.index_path_, f'sample.{r}.pt')
            sub_sample = torch.load(sub_sample_path)
            os.remove(sub_sample_path)
            endpos = offset + sub_sample.size(0)
            sample[offset:endpos] = sub_sample
            offset = endpos
        
        sample = sample[torch.randperm(sample.size(0))]
        heldout_size = int(min(0.05 * sample.size(0), 50_000))
        return sample.split([sample.size(0) - heldout_size, heldout_size])

    def _train_kmeans(self, sample, shared_lists):
        """使用 FAISS 在 GPU 上高效地训练 K-means。"""
        torch.cuda.empty_cache()
        # 将样本数据放入共享列表，以便传递给另一个进程
        shared_lists[0].append(sample)
        return_value_queue = mp.Queue()
        
        args_ = [self.config.dim, self.num_partitions, self.config.kmeans_niters, shared_lists, return_value_queue]
        proc = mp.Process(target=compute_faiss_kmeans, args=args_)
        proc.start()
        centroids = return_value_queue.get()
        proc.join()
        
        return torch.nn.functional.normalize(centroids, dim=-1).half()

    def _compute_avg_residual(self, centroids, heldout):
        """使用 heldout 集计算残差量化的桶边界和权重。"""
        compressor = ResidualCodec(config=self.config, centroids=centroids)
        codes = compressor.compress_into_codes(heldout, out_device='cuda')
        reconstructed = compressor.lookup_centroids(codes, out_device='cuda')
        residuals = heldout.cuda() - reconstructed

        avg_residual = torch.abs(residuals).mean()

        # 计算分位数以确定桶的边界和权重
        num_options = 2 ** self.config.nbits
        quantiles = torch.arange(1, num_options, device=residuals.device) / num_options
        bucket_cutoffs = residuals.float().quantile(quantiles)
        
        # 桶权重取每个分位区间的中点
        bucket_weights_quantiles = quantiles - (0.5 / num_options)
        bucket_weights = residuals.float().quantile(bucket_weights_quantiles)

        return bucket_cutoffs, bucket_weights, avg_residual.item()

    def index(self):
        """索引阶段：所有进程并行地对自己分到的文档块进行编码和保存。"""
        with self.saver.thread():
            batches = self.collection.enumerate_batches(rank=self.rank)
            for chunk_idx, offset, passages in tqdm.tqdm(batches, disable=self.rank > 0):
                embs, doclens = self.encoder.encode_passages(passages)
                Run().print_main(f"#> 正在保存块 {chunk_idx}: {len(passages):,} 个段落, {embs.size(0):,} 个嵌入。")
                self.saver.save_chunk(chunk_idx, offset, embs, doclens)

    def finalize(self):
        """最终化阶段：仅由主进程执行，合并索引信息，构建 IVF。"""
        if self.rank > 0:
            return

        self._collect_embedding_id_offset()
        self._build_ivf()
        self._update_metadata()

    def _collect_embedding_id_offset(self):
        """收集每个索引块的元数据，并计算全局的嵌入偏移量。"""
        passage_offset, embedding_offset = 0, 0
        self.embedding_offsets = []
        for chunk_idx in range(self.num_chunks):
            metadata_path = os.path.join(self.config.index_path_, f'{chunk_idx}.metadata.json')
            with open(metadata_path) as f:
                chunk_metadata = ujson.load(f)
                chunk_metadata['embedding_offset'] = embedding_offset
                self.embedding_offsets.append(embedding_offset)
                passage_offset += chunk_metadata['num_passages']
                embedding_offset += chunk_metadata['num_embeddings']
            with open(metadata_path, 'w') as f:
                ujson.dump(chunk_metadata, f, indent=4)
        self.num_embeddings = embedding_offset

    def _build_ivf(self):
        """构建倒排文件 (IVF)，这是实现快速候选检索的核心。"""
        codes = torch.empty(self.num_embeddings, dtype=torch.int32)
        # 从所有块中加载 codes
        for chunk_idx in range(self.num_chunks):
            offset = self.embedding_offsets[chunk_idx]
            chunk_codes = ResidualCodec.Embeddings.load_codes(self.config.index_path_, chunk_idx)
            codes[offset:offset+chunk_codes.size(0)] = chunk_codes
        
        # 对 codes 进行排序，得到每个分区的嵌入ID列表
        indices, values = codes.sort()
        partitions, ivf_lengths = values.unique_consecutive(return_counts=True)
        assert partitions.size(0) == self.num_partitions

        torch.save((indices, ivf_lengths), os.path.join(self.config.index_path_, 'ivf.pt'))

    def _update_metadata(self):
        """保存最终的索引元数据文件 (metadata.json)。"""
        metadata_path = os.path.join(self.config.index_path_, 'metadata.json')
        Run().print("#> 正在将最终索引元数据保存至", metadata_path, "..")
        with open(metadata_path, 'w') as f:
            d = {'config': self.config.export(), 'num_chunks': self.num_chunks,
                 'num_partitions': self.num_partitions, 'num_embeddings': self.num_embeddings,
                 'avg_doclen': self.num_embeddings / len(self.collection)}
            f.write(ujson.dumps(d, indent=4) + '\n')


def compute_faiss_kmeans(dim, num_partitions, kmeans_niters, shared_lists, return_value_queue=None):
    """一个独立的函数，用于在子进程中运行 FAISS K-means，以隔离其 GPU 显存使用。"""
    kmeans = faiss.Kmeans(dim, num_partitions, niter=kmeans_niters, gpu=True, verbose=True, seed=123)
    sample = shared_lists[0][0].float().numpy()
    kmeans.train(sample)
    centroids = torch.from_numpy(kmeans.centroids)
    if return_value_queue:
        return_value_queue.put(centroids)
    return centroids