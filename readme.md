### 📂 项目源码结构

这是一个结合了 **ColBERT** 检索框架和 **Baleen** 多跳问答应用的项目结构。ColBERT 提供了高效的“晚期交互”检索核心，而 Baleen 则在此基础上构建了更复杂的、逐步推理的问答引擎。

------

#### 📦 `baleen/`



> `baleen` 包，实现了基于 ColBERT 的高级多跳（multi-hop）问答引擎。

- `__init__.py`: 将 `baleen` 目录声明为 Python 的一个包。
- `engine.py`: 定义了 `Baleen` 引擎，它是多跳问答流程的总控制器，协调检索和信息精炼。
- `hop_searcher.py`: 定义了 `HopSearcher`，一个为多跳搜索定制的、支持上下文感知的 `Searcher` 子类。
- **`condenser/`**: 包含“信息冷凝器” (Condenser) 组件，负责从检索到的段落中精炼出关键信息。
  - `__init__.py`: 将 `condenser` 目录声明为一个子包。
  - `condense.py`: 实现了 `Condenser` 类的核心逻辑，通过两阶段过程从段落中抽取关键句子。
  - `model.py`: 定义了 Condenser 使用的底层 `ElectraReader` 模型，用于对句子进行相关性评分。
  - `tokenization.py`: 为 Condenser 模型定义的专用分词器，处理查询和候选句子的拼接。
- **`utils/`**: 包含 `baleen` 应用所需的辅助工具。
  - `__init__.py`: 将 `utils` 目录声明为一个子包。
  - `annotate.py`: 提供了一个为排序列表自动添加标签（判断是否为正确答案）的脚本。
  - `loaders.py`: 包含为 `baleen` 加载特定数据格式（如上下文或按句子切分的集合）的函数。

------

#### 🚀 `colbert/`



> `colbert` 核心框架包，包含了从数据处理、模型定义、训练、索引到搜索的全部核心功能。

- `__init__.py`: 导入 `colbert` 包的顶层 API，如 `Trainer`, `Indexer`, `Searcher`。
- `indexer.py`: 定义了 `Indexer` 类，这是用于构建 ColBERT 索引的用户友好顶层接口。
- `searcher.py`: 定义了 `Searcher` 类，这是用于从索引中进行检索的用户友好顶层接口。
- `trainer.py`: 定义了 `Trainer` 类，这是用于启动和管理模型训练过程的用户友好顶层接口。
- `parameters.py`: 定义了项目中使用的一些全局常量，如默认设备（DEVICE）和需要保存的检查点列表。
- **`data/`**: 负责所有数据的抽象和加载。
  - `__init__.py`: 导入 `Collection`, `Queries` 等所有数据处理类。
  - `collection.py`: 定义 `Collection` 类，用于表示和操作文档（段落）集合。
  - `examples.py`: 定义 `Examples` 类，用于处理训练样本（特别是三元组）。
  - `queries.py`: 定义 `Queries` 类，用于加载和管理查询（问题）集合。
  - `ranking.py`: 定义 `Ranking` 类，用于处理和表示模型的排序结果。
  - `dataset.py`: 一个占位符文件，计划用于定义一个更综合的数据集类。
- **`evaluation/`**: 包含与模型评估相关的工具和脚本。
  - `__init__.py`: 将 `evaluation` 目录声明为一个子包。
  - `load_model.py`: 提供了加载预训练 ColBERT 模型及其权重的核心函数。
  - `loaders.py`: 包含一系列用于加载评估所需数据（如查询、qrels、文档集合）的函数。
  - `metrics.py`: 定义 `Metrics` 类，用于计算、记录和保存 MRR、Recall 等标准检索指标。
- **`indexing/`**: 负责将文档集合转换为高效的 ColBERT 索引。
  - `__init__.py`: 将 `indexing` 目录声明为一个子包。
  - `collection_encoder.py`: 定义 `CollectionEncoder` 类，负责将文本段落批量编码为词元级嵌入。
  - `collection_indexer.py`: 定义 `CollectionIndexer` 类，是索引构建流程的核心协调器。
  - `index_manager.py`: 定义了一个简单的 `IndexManager` 类，用于管理索引文件的保存。
  - `index_saver.py`: 定义 `IndexSaver` 类，以多线程方式将索引块异步写入磁盘。
  - `loaders.py`: 提供了一些用于从索引目录中加载特定部分文件（如文档长度）的辅助函数。
  - **`codecs/`**: 包含用于向量压缩和解压缩的编解码器。
    - `__init__.py`: 将 `codecs` 目录声明为一个子包。
    - `residual.py`: 定义 `ResidualCodec` 类，实现了基于残差量化的核心压缩算法。
    - `residual_embeddings.py`: 定义 `ResidualEmbeddings` 类，一个用于存储压缩后嵌入的数据容器。
    - `residual_embeddings_strided.py`: 在 `ResidualEmbeddings` 基础上增加了对 `StridedTensor` 的支持，用于快速按文档ID查找。
- **`infra/`**: 项目的基础设施层，负责配置、启动和运行管理。
  - `__init__.py`: 导入 `Run` 和配置类，方便上层调用。
  - `launcher.py`: 定义 `Launcher` 类，是实现分布式（多 GPU）计算的核心，负责启动和管理多进程任务。
  - `provenance.py`: 定义 `Provenance` 类，用于追踪对象的创建过程（调用堆栈），增强实验的可追溯性。
  - `run.py`: 定义 `Run` 全局单例，用于管理整个实验的生命周期和上下文。
  - **`config/`**: 负责项目的所有配置管理。
    - `__init__.py`: 导入所有配置类。
    - `core_config.py`: 定义了配置类的核心基类 `CoreConfig`，提供了处理默认值、动态配置等基础功能。
    - `base_config.py`: 继承 `CoreConfig` 并提供了从文件、检查点等来源加载配置的高级功能。
    - `settings.py`: 使用 `dataclasses` 定义了所有具体的配置项，按功能（训练、索引、搜索等）分组。
    - `config.py`: 定义了最终的 `ColBERTConfig` 和 `RunConfig` 类，它们通过多重继承将所有配置项组合在一起。
  - **`utilities/`**: 包含了一些封装好的高级工作流脚本。
    - `__init__.py`: 将 `utilities` 目录声明为一个子包。
    - `annotate_em.py`: 定义 `AnnotateEM` 类，用于对检索结果进行精确匹配（EM）的自动标注。
    - `create_triples.py`: 定义 `Triples` 类，用于从已标注的排序列表中生成训练三元组。
    - `minicorpus.py`: 提供了一个用于从大规模数据集中采样创建“迷你”语料库的脚本。
- **`modeling/`**: ColBERT 模型的核心，包含架构定义、分词和推理接口。
  - `__init__.py`: 导入 `tokenization` 模块中的主要类。
  - `base_colbert.py`: 定义 `BaseColBERT` 类，一个基础的、封装了模型、配置和分词器的模块。
  - `checkpoint.py`: 定义 `Checkpoint` 类，一个专门用于推理（Inference）的便捷接口。
  - `colbert.py`: 实现了 ColBERT 模型的核心架构和 `forward` 训练逻辑，包括 MaxSim 评分。
  - `hf_colbert.py`: 定义了 `HF_ColBERT` 类，它继承自 HuggingFace 的 `BertPreTrainedModel`，实现了与 Transformers 库的无缝集成。
  - **`reranker/`**: 包含用于交叉注意力重排序的模型。
    - `__init__.py`: 将 `reranker` 目录声明为一个子包。
    - `electra.py`: 定义了基于 `Electra` 模型的重排序器 `ElectraReranker`。
    - `tokenizer.py`: 为重排序器定义的专用分词器，负责将查询和文档拼接为单一序列。
  - **`tokenization/`**: ColBERT 的专用分词和编码逻辑。
    - `__init__.py`: 导入所有分词器类。
    - `doc_tokenization.py`: 定义 `DocTokenizer`，负责处理文档，并添加特殊的 `[D]` 标记。
    - `query_tokenization.py`: 定义 `QueryTokenizer`，负责处理查询，添加 `[Q]` 标记并使用 `[MASK]` 进行查询增强。
    - `utils.py`: 包含将三元组文本数据转换为模型所需张量格式的辅助函数。
- **`search/`**: 实现了 ColBERT 高效检索的后端逻辑。
  - `__init__.py`: 将 `search` 目录声明为一个子包。
  - `candidate_generation.py`: 实现了检索的第一阶段：利用 IVF 索引快速生成候选文档集。
  - `index_loader.py`: 定义 `IndexLoader` 类，负责从磁盘加载 ColBERT 索引的各个组成部分。
  - `index_storage.py`: 定义 `IndexScorer` 类，整合了索引加载、候选生成和最终评分功能，是 `Searcher` 的核心工作引擎。
  - `strided_tensor.py`: 定义 `StridedTensor` 的高级接口，提供了便捷的 `lookup` 方法。
  - `strided_tensor_core.py`: `StridedTensor` 的核心实现，利用 `torch.as_strided` 创建高效的内存视图。
- **`training/`**: 包含了模型训练的所有核心逻辑。
  - `__init__.py`: 将 `training` 目录声明为一个子包。
  - `training.py`: 包含了核心的 `train` 函数，实现了训练循环、损失计算、优化器步骤和检查点管理。
  - `lazy_batcher.py`: 定义了 `LazyBatcher`，一种高效的数据加载策略，在需要时才“懒加载”文本内容。
  - `rerank_batcher.py`: 为交叉注意力的重排序器模型定义的专用数据加载器。
  - `eager_batcher.py`: 定义了 `EagerBatcher`，一种较旧的数据加载策略，一次性加载数据到内存。
  - `utils.py`: 包含了训练过程中使用的一些辅助函数，如打印进度和管理检查点。
- **`utils/`**: 提供了贯穿整个项目的通用工具函数和辅助类。
  - `__init__.py`: 将 `utils` 目录声明为一个子包。
  - `amp.py`: 定义 `MixedPrecisionManager` 类，用于简化 PyTorch 自动混合精度训练的流程。
  - `distributed.py`: 提供了初始化和管理 PyTorch 分布式训练环境的辅助函数。
  - `logging.py`: 定义 `Logger` 类，用于处理实验过程中的日志记录。
  - `parser.py`: 提供 `Arguments` 类，用于封装 `argparse`，简化命令行参数的定义和解析。
  - `runs.py`: 定义了 `_RunManager` 类，是 `infra/run.py` 的一个早期版本，用于全局管理实验运行。
  - `utils.py`: 包含了一个各种通用辅助函数的集合，被项目的许多其他部分所使用。

------

### 🛠 `utility/`



> `utility` 包，包含了一系列不属于核心框架但对实验流程非常重要的辅助脚本和工具。

- `__init__.py`: 将 `utility` 目录声明为一个包。
- **`evaluate/`**: 包含用于对检索结果进行自动化评测和标注的脚本。
  - `__init__.py`: 将 `evaluate` 目录声明为一个子包。
  - `annotate_EM.py`: 一个对检索结果进行精确匹配（Exact Match）自动标注的脚本。
  - `annotate_EM_helpers.py`: 包含了 `annotate_EM.py` 脚本所调用的核心辅助函数。
  - `msmarco_passages.py`: 一个专门用于评估 MS MARCO Passages 数据集上排序结果的脚本。
- **`preprocess/`**: 包含数据预处理脚本。
  - `__init__.py`: 将 `preprocess` 目录声明为一个子包。
  - `docs2passages.py`: 一个将长文档集合切分为固定长度段落的脚本。
  - `queries_split.py`: 一个将查询文件随机分割成两个子集（如训练集和验证集）的脚本。
- **`rankings/`**: 包含用于后处理排序结果文件的脚本。
  - `__init__.py`: 将 `rankings` 目录声明为一个子包。
  - `dev_subsample.py`: 用于从一个大的排序结果文件中，根据一个小的查询子集抽取出对应的排序结果。
  - `merge.py`: 用于合并多个排序文件，并根据分数重新排序。
  - `split_by_offset.py`: 根据 QID 的特殊偏移量，将一个合并后的排序文件拆分成多个原始文件。
  - `split_by_queries.py`: 根据一个或多个查询文件，将一个大的排序结果文件拆分成对应的部分。
  - `tune.py`: 一个自动从一系列评估结果中找出最佳模型检查点的脚本。
- **`supervision/`**: 包含用于自动生成训练数据的脚本。
  - `__init__.py`: 将 `supervision` 目录声明为一个子包。
  - `self_training.py`: 通过自训练（假设排名靠前为正、靠后为负）从无标注排序列表中生成训练三元组。
  - `triples.py`: 从已标注的排序列表中采样生成高质量训练三元组的核心脚本。
- **`utils/`**: 包含了 `utility` 包内部使用的更基础的辅助函数。
  - `__init__.py`: 将 `utils` 目录声明为一个子包。
  - `dpr.py`: 包含了一些源自 DPR 项目的文本处理函数，用于实现兼容的精确匹配（EM）评测。
  - `qa_loaders.py`: 提供了加载特定格式的问答（QA）数据和文档集合的辅助函数。
  - `save_metadata.py`: 提供了用于捕获和保存实验元数据（如 Git 哈希、命令行参数）的函数。

