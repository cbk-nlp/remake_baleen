# 文件名: colbert/evaluation/load_model.py

# 导入 ColBERT 模型定义
from colbert.modeling.colbert import ColBERT
# 导入 PyTorch 设备参数 (例如, 'cuda')
from colbert.parameters import DEVICE
# 导入工具函数，用于加载模型检查点和打印格式化信息
from colbert.utils.utils import print_message, load_checkpoint


def load_model(args, do_print=True):
    """
    加载一个预训练的 ColBERT 模型及其检查点权重。

    该函数首先会根据传入的参数（args）初始化一个 ColBERT 模型结构，
    然后从指定的检查点路径加载训练好的权重，并将其设置为评估模式。

    Args:
        args (Namespace): 一个包含模型配置参数的对象。
                          需要包含 query_maxlen, doc_maxlen, dim, similarity,
                          mask_punctuation, 和 checkpoint 等属性。
        do_print (bool, optional): 是否打印加载过程中的信息。默认为 True。

    Returns:
        tuple:
            - colbert (ColBERT): 加载了权重并处于评估模式的 ColBERT 模型实例。
            - checkpoint (dict): 从检查点文件中加载的完整检查点对象，包含模型状态、优化器状态和训练参数等。
    """
    # 基于 'bert-base-uncased' 结构和传入的参数初始化 ColBERT 模型
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)
    # 将模型移动到指定的设备 (例如, GPU)
    colbert = colbert.to(DEVICE)

    # 打印加载信息
    print_message("#> 正在加载模型检查点...", condition=do_print)

    # 从 args.checkpoint 路径加载模型权重
    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    # 将模型设置为评估模式，这会关闭 dropout 等仅在训练时使用的层
    colbert.eval()

    return colbert, checkpoint