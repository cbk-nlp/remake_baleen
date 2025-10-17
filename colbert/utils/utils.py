# 文件名: colbert/utils/utils.py

import os
import tqdm
import torch
import datetime
import itertools
from collections import OrderedDict, defaultdict


def print_message(*s, condition=True, pad=False):
    """
    打印带有时间戳的格式化消息。

    Args:
        *s: 任意数量的要打印的对象，它们会被转换成字符串并用空格连接。
        condition (bool, optional): 只有当此条件为 True 时才打印消息。默认为 True。
        pad (bool, optional): 是否在消息前后添加空行。默认为 False。
    
    Returns:
        str: 格式化后的消息字符串。
    """
    s = ' '.join([str(x) for x in s])
    msg = f"[{datetime.datetime.now().strftime('%b %d, %H:%M:%S')}] {s}"

    if condition:
        # 如果需要，添加垂直填充
        msg_to_print = f'\n{msg}\n' if pad else msg
        print(msg_to_print, flush=True)

    return msg


def timestamp(daydir=False):
    """
    生成一个当前时间的字符串，可用作文件名或目录名。

    Args:
        daydir (bool, optional): 如果为 True，则在日期和时间之间使用斜杠'/'，
                                 适用于创建分层目录。默认为 False。
    
    Returns:
        str: 格式化后的时间戳字符串。
    """
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    return datetime.datetime.now().strftime(format_str)


def file_tqdm(file):
    """
    为文件读取创建一个 `tqdm` 进度条，以 MiB 为单位显示进度。

    Args:
        file (file object): 已打开的文件对象。

    Yields:
        str: 文件中的每一行。
    """
    print(f"#> 正在读取 {file.name}")
    total_size_mib = os.path.getsize(file.name) / 1024.0 / 1024.0
    with tqdm.tqdm(total=total_size_mib, unit="MiB", desc=f"读取文件 {os.path.basename(file.name)}") as pbar:
        for line in file:
            yield line
            # 根据读取的字节数更新进度条
            pbar.update(len(line.encode('utf-8')) / 1024.0 / 1024.0)

def torch_load_dnn(path):
    """
    从本地路径或 URL 加载一个 PyTorch 模型文件（特别是为旧的 .dnn 格式设计）。
    总是将模型加载到 CPU。
    """
    if path.startswith("http:") or path.startswith("https:"):
        return torch.hub.load_state_dict_from_url(path, map_location='cpu')
    return torch.load(path, map_location='cpu')


def save_checkpoint(path, epoch_idx, mb_idx, model, optimizer, arguments=None):
    """(较旧的函数) 保存一个完整的训练检查点。"""
    print(f"#> 正在保存检查点至 {path} ...")
    # 如果模型被 DistributedDataParallel 包装，则获取内部的模型
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {
        'epoch': epoch_idx,
        'batch': mb_idx,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'arguments': arguments
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, checkpoint=None, optimizer=None, do_print=True):
    """
    (较旧的函数) 加载一个训练检查点到模型和优化器中。
    它会自动处理 `module.` 前缀，这是由 `torch.nn.DataParallel` 或
    `torch.nn.parallel.DistributedDataParallel` 添加的。
    """
    if do_print:
        print_message("#> 正在从", path, "加载检查点...")
    
    if checkpoint is None:
        checkpoint = load_checkpoint_raw(path)

    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError:
        # 如果严格加载失败（例如，由于模型结构有细微变化），则尝试非严格加载
        print_message("[警告] 使用 strict=False 模式加载检查点。")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if do_print:
        print_message(f"#> 检查点信息: epoch = {checkpoint.get('epoch', 'N/A')}, batch = {checkpoint.get('batch', 'N/A')}")
    return checkpoint


def load_checkpoint_raw(path):
    """加载原始检查点文件并处理 'module.' 前缀。"""
    checkpoint = torch_load_dnn(path)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    checkpoint['model_state_dict'] = new_state_dict
    return checkpoint


def create_directory(path):
    """创建目录，如果它不存在的话。如果存在，则打印一条提示信息。"""
    if os.path.exists(path):
        print_message("#> 提示: 输出目录", path, '已存在。')
    else:
        print_message("#> 正在创建目录", path)
        os.makedirs(path, exist_ok=True)

def f7(seq):
    """
    从一个序列中移除重复项，同时保持原始顺序。
    来源: https://stackoverflow.com/a/480227/1493011
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def batch(group, bsize, provide_offset=False):
    """
    将一个列表或元组分割成多个指定大小的批次。

    Args:
        group (list or tuple): 待分割的序列。
        bsize (int): 每个批次的大小。
        provide_offset (bool, optional): 如果为 True，则每个批次返回 `(offset, batch_content)`。
                                        默认为 False。
    Yields:
        list or tuple: 一个批次的数据。
    """
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield (offset, L) if provide_offset else L
        offset += len(L)


class dotdict(dict):
    """
    一个字典子类，允许通过点表示法 (d.key) 来访问字典的键值，
    类似于 JavaScript 中的对象属性访问。
    来源: https://stackoverflow.com/questions/2352181
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class dotdict_lax(dict):
    """dotdict 的一个宽松版本，如果键不存在，`__getattr__` 会返回 None 而不是抛出 KeyError。"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten(L):
    """将一个嵌套列表（列表的列表）扁平化成一个单一的列表。"""
    return [item for sublist in L for item in sublist]


def zipstar(L, lazy=False):
    """
    一个更快的 `zip(*L)` 实现，用于解压一个列表的列表。
    例如, `zipstar([(a1, b1), (a2, b2)])` -> `[[a1, a2], [b1, b2]]`
    """
    if not L:
        return []
    
    # 对于较窄的列表，列表推导式更快
    width = len(L[0])
    if width < 100:
        return [[elem[idx] for elem in L] for idx in range(width)]

    # 对于较宽的列表，`zip` 更高效
    result = zip(*L)
    return result if lazy else list(result)


def grouper(iterable, n, fillvalue=None):
    """
    将一个可迭代对象按固定长度 n 进行分组。
    示例: grouper('ABCDEFG', 3, 'x') -> ABC DEF Gxx
    来源: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def lengths2offsets(lengths):
    """
    将一个长度列表转换为一个偏移量元组的生成器。
    示例: [2, 3, 1] -> (0, 2), (2, 5), (5, 6)
    """
    offset = 0
    for length in lengths:
        yield (offset, offset + length)
        offset += length


class NullContextManager(object):
    """
    一个空的上下文管理器，它不做任何事情。
    在需要一个上下文管理器但又不需要实际功能时很有用，
    例如在 `amp.py` 中用于禁用混合精度。
    """
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


def load_batch_backgrounds(args, qids):
    """
    (特定于某些实验的函数)
    根据查询ID (qids) 从 `args.qid2backgrounds` 中加载背景/上下文信息。
    """
    if not hasattr(args, 'qid2backgrounds') or args.qid2backgrounds is None:
        return None

    qbackgrounds = []
    for qid in qids:
        back = args.qid2backgrounds[qid]
        if back:
            # 根据背景是 PID 还是其他格式来加载文本
            if isinstance(back[0], int):
                x = [args.collection[pid] for pid in back]
            else:
                x = [args.collectionX.get(pid, '') for pid in back]
            qbackgrounds.append(' [SEP] '.join(x))
        else:
            qbackgrounds.append('')
    
    return qbackgrounds