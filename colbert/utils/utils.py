# 文件名: colbert/utils/utils.py

import os
import tqdm
import torch
import datetime
import itertools

from multiprocessing import Pool
from collections import OrderedDict, defaultdict


def print_message(*s, condition=True, pad=False):
    """
    打印带有时间戳的消息。

    参数:
        *s: 任意数量的参数，将被转换成字符串并用空格连接。
        condition (bool): 只有当此条件为 True 时才打印消息。
        pad (bool): 如果为 True，则在消息前后添加换行符。

    返回:
        str: 格式化后的消息字符串（无论是否打印）。
    """
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        msg = msg if not pad else f'\n{msg}\n'
        print(msg, flush=True)

    return msg


def timestamp(daydir=False):
    """
    生成一个当前时间戳字符串。

    参数:
        daydir (bool): 如果为 True，使用斜杠 '/' 作为日期分隔符（适用于目录）；
                       否则使用短横线 '-' 和下划线 '_'。

    返回:
        str: 格式化的时间戳字符串。
    """
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


def file_tqdm(file):
    """
    使用 tqdm 为文件读取提供一个进度条（单位：MiB）。

    参数:
        file: 一个打开的文件对象。

    返回:
        generator: 逐行产生文件的内容。
    """
    print(f"#> Reading {file.name}")

    # 计算文件总大小（MiB）
    with tqdm.tqdm(total=os.path.getsize(file.name) / 1024.0 / 1024.0, unit="MiB") as pbar:
        for line in file:
            yield line
            # 更新进度条
            pbar.update(len(line) / 1024.0 / 1024.0)

        pbar.close()


def torch_load_dnn(path):
    """
    从本地路径或 URL 加载 PyTorch 模型（DNN）的状态字典。
    所有张量将被加载到 CPU。

    参数:
        path (str): 模型文件路径或 URL。

    返回:
        dict: PyTorch 模型的 state_dict。
    """
    if path.startswith("http:") or path.startswith("https:"):
        # 从 URL 加载
        dnn = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        # 从本地路径加载
        dnn = torch.load(path, map_location='cpu')
    
    return dnn

def save_checkpoint(path, epoch_idx, mb_idx, model, optimizer, arguments=None):
    """
    保存训练检查点（checkpoint）到指定路径。

    参数:
        path (str): 保存路径。
        epoch_idx (int): 当前的 epoch 索引。
        mb_idx (int): 当前的 batch (minibatch) 索引。
        model (torch.nn.Module): 要保存的模型。
        optimizer (torch.optim.Optimizer): 要保存的优化器。
        arguments (any, optional): 
            需要一同保存的其他参数（例如命令行参数）。
    """
    print(f"#> Saving a checkpoint to {path} ..")

    if hasattr(model, 'module'):
        model = model.module  # 处理 DataParallel 或 DDP 包装的模型

    checkpoint = {}
    checkpoint['epoch'] = epoch_idx
    checkpoint['batch'] = mb_idx
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    checkpoint['arguments'] = arguments

    torch.save(checkpoint, path)


def load_checkpoint(path, model, checkpoint=None, optimizer=None, do_print=True):
    """
    加载检查点（checkpoint）到模型和优化器中。

    参数:
        path (str): 检查点文件路径（如果 checkpoint 参数为 None）。
        model (torch.nn.Module): 目标模型。
        checkpoint (dict, optional): 
            预加载的检查点字典。如果为 None，则从 path 加载。
        optimizer (torch.optim.Optimizer, optional): 
            目标优化器（如果需要恢复优化器状态）。
        do_print (bool): 是否打印加载信息。

    返回:
        dict: 加载的检查点字典。
    """
    if do_print:
        print_message("#> Loading checkpoint", path, "..")

    if checkpoint is None:
        # 如果未提供 checkpoint 字典，则从路径加载
        checkpoint = load_checkpoint_raw(path)

    try:
        # 尝试严格加载
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        # 如果键不完全匹配，尝试非严格加载
        print_message("[WARNING] Loading checkpoint with strict=False")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if optimizer:
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if do_print:
        print_message("#> checkpoint['epoch'] =", checkpoint['epoch'])
        print_message("#> checkpoint['batch'] =", checkpoint['batch'])

    return checkpoint


def load_checkpoint_raw(path):
    """
    从路径或 URL 加载原始检查点文件，并清理模型状态字典的键。
    (主要用于去除 'module.' 前缀，以便在非并行模式下加载)

    参数:
        path (str): 检查点文件路径或 URL。

    返回:
        dict: 清理（键名）后的检查点字典。
    """
    if path.startswith("http:") or path.startswith("https:"):
        checkpoint = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location='cpu')

    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k.startswith('module.'):
            # 移除 'module.' 前缀
            name = k[7:]
        new_state_dict[name] = v

    checkpoint['model_state_dict'] = new_state_dict

    return checkpoint


def create_directory(path):
    """
    创建指定的目录（如果它还不存在）。

    参数:
        path (str): 要创建的目录路径。
    """
    if os.path.exists(path):
        print('\n')
        print_message("#> Note: Output directory", path, 'already exists\n\n')
    else:
        print('\n')
        print_message("#> Creating directory", path, '\n\n')
        os.makedirs(path)

# def batch(file, bsize):
#     """
#     (此函数已被注释掉)
#     从文件中批量读取 bsize 行 ujson 数据。
#     """
#     while True:
#         L = [ujson.loads(file.readline()) for _ in range(bsize)]
#         yield L
#     return


def f7(seq):
    """
    从序列中移除重复项，同时保持原始顺序。
    (来源: https://stackoverflow.com/a/480227/1493011)

    参数:
        seq (list): 输入的序列。

    返回:
        list: 移除了重复项的列表。
    """
    seen = set()
    # seen.add(x) 总会返回 None (False)，
    # 所以 'x in seen or seen.add(x)' 
    # 会在 x 第一次出现时（not in seen）执行 seen.add(x)
    return [x for x in seq if not (x in seen or seen.add(x))]


def batch(group, bsize, provide_offset=False):
    """
    将一个列表（group）分割成指定大小（bsize）的批次（batch）。

    参数:
        group (list): 要分割的列表。
        bsize (int): 每个批次的大小。
        provide_offset (bool): 
            如果为 True，yield (offset, batch)；
            否则，yield batch。

    返回:
        generator: 产生批次的生成器。
    """
    offset = 0
    while offset < len(group):
        L = group[offset: offset + bsize]
        yield ((offset, L) if provide_offset else L)
        offset += len(L)
    return


class dotdict(dict):
    """
    一个支持“点表示法”（dot.notation）访问的字典子类。
    允许使用 d.key 代替 d['key']。
    (来源: derek73 @ https://stackoverflow.com/questions/2352181)
    """
    # 允许通过 d.key 读取
    __getattr__ = dict.__getitem__
    # 允许通过 d.key = v 设置
    __setattr__ = dict.__setitem__
    # 允许通过 del d.key 删除
    __delattr__ = dict.__delitem__


class dotdict_lax(dict):
    """
    一个“宽松”版的 dotdict。
    当访问不存在的键时，返回 None 而不是抛出 KeyError (因为它使用 dict.get)。
    """
    # 允许通过 d.key 读取 (宽松)
    __getattr__ = dict.get
    # 允许通过 d.key = v 设置
    __setattr__ = dict.__setitem__
    # 允许通过 del d.key 删除
    __delattr__ = dict.__delitem__


def flatten(L):
    """
    将一个嵌套列表（list of lists）“展平”成一个单一列表。

    参数:
        L (list): 嵌套列表，例如 [[1, 2], [3, 4]]。

    返回:
        list: 展平后的列表，例如 [1, 2, 3, 4]。
    """
    # return [x for y in L for x in y] # 列表推导式版本

    result = []
    for _list in L:
        result += _list

    return result


def zipstar(L, lazy=False):
    """
    功能上等同于 zip(*L)，即“转置”一个列表的列表。
    据称比 Python 内置的 zip(*L) 更快，尤其是在宽度（内部列表长度）较小时。

    参数:
        L (list): 输入的列表，例如 [(a, b, c), (a, b, c), ...]。
        lazy (bool): 如果为 True 且宽度 >= 100，返回一个迭代器；
                     否则返回一个列表。

    返回:
        list or iterator: 转置后的结果，例如 [[a, a, ...], [b, b, ...], [c, c, ...]]。
    """

    if len(L) == 0:
        return L

    width = len(L[0])

    if width < 100:
        # 当宽度较小时，使用列表推导式
        return [[elem[idx] for elem in L] for idx in range(width)]

    # 当宽度较大时，使用内置的 zip
    L = zip(*L)

    return L if lazy else list(L)


def zip_first(L1, L2):
    """
    打包（zip）两个列表 L1 和 L2。
    包含一个断言，以确保 L1（如果是列表或元组）的长度与打包后的长度一致。

    参数:
        L1: 第一个可迭代对象。
        L2: 第二个可迭代对象。

    返回:
        list: 打包后的元组列表。
    """
    length = len(L1) if type(L1) in [tuple, list] else None

    L3 = list(zip(L1, L2))

    # 断言：确保 zip 操作没有因为 L1 和 L2 长度不等而截断 L1
    assert length in [None, len(L3)], "zip_first() failure: length differs!"

    return L3


def int_or_float(val):
    """
    根据字符串是否包含小数点，将其转换为 int 或 float。

    参数:
        val (str): 输入的字符串值。

    返回:
        int or float: 转换后的数值。
    """
    if '.' in val:
        return float(val)
        
    return int(val)

def load_ranking(path, types=None, lazy=False):
    """
    从文件加载一个排名（ranking）列表。
    首先尝试将其作为 PyTorch tensor (torch.load) 加载。
    如果失败，则假定它是一个文本文件（例如 TSV），
    并逐行读取，使用 'types' 中的函数转换每一列。

    参数:
        path (str): 排名文件的路径。
        types (list/cycle, optional): 
            一个类型/函数列表，用于转换文本文件的每一列。
        lazy (bool): (仅当 torch.load 成功时) 传递给 zipstar。

    返回:
        list: 加载的排名列表。
    """
    print_message(f"#> Loading the ranked lists from {path} ..")

    try:
        # 尝试作为 PyTorch 文件加载
        lists = torch.load(path)
        lists = zipstar([l.tolist() for l in tqdm.tqdm(lists)], lazy=lazy)
    except:
        # 如果失败，作为文本文件加载
        if types is None:
            # 默认类型为 int_or_float
            types = itertools.cycle([int_or_float])

        with open(path) as f:
            lists = [[typ(x) for typ, x in zip_first(types, line.strip().split('\t'))]
                     for line in file_tqdm(f)]

    return lists


def save_ranking(ranking, path):
    """
    将排名（ranking）列表使用 torch.save 保存到文件。
    在保存前，它会先转置 ranking 并将内部列表转换为 tensor。

    参数:
        ranking (list): 排名列表。
        path (str): 保存路径。

    返回:
        list: 转换（转置并转为 tensor）后的列表。
    """
    # 转置 ranking
    lists = zipstar(ranking)
    # 转换为 tensor
    lists = [torch.tensor(l) for l in lists]

    torch.save(lists, path)

    return lists


def groupby_first_item(lst):
    """
    根据子列表的第一个元素对列表进行分组。
    假定列表中的项是 [first_item, item1, item2, ...] 或 [first_item, item]。

    参数:
        lst (list): 
            形如 [[key1, v1], [key1, v2], [key2, v3], ...] 的列表。

    返回:
        defaultdict: 以 'first_item' 为键，
                     包含剩余元素列表（[item, ...] 或 item）为值的字典。
    """
    groups = defaultdict(list)

    for first, *rest in lst:
        # 如果 rest 只有一个元素，解包；否则保持为列表
        rest = rest[0] if len(rest) == 1 else rest
        groups[first].append(rest)

    return groups


def process_grouped_by_first_item(lst):
    """
    处理一个 *已经* 按照第一个元素分组（或排序）的列表。
    这是一个生成器，当第一个元素（key）发生变化时，
    它会 yield 上一个 (key, values) 对。

    *注意: 此函数要求输入列表 'lst' 必须已按第一个元素排序。*

    参数:
        lst (list): 
            已按 key 排序的列表，例如 [[k1, v1], [k1, v2], [k2, v3], ...]

    返回:
        generator or defaultdict: 
            产生 (key, values) 对的生成器。
            (注：此函数的实现在最后返回了 groups，
             但这可能只包含最后一组数据，主要用途是 'yield' 部分。)
    """
    groups = defaultdict(list)

    started = False
    last_group = None

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest

        if started and first != last_group:
            # 当 key 变化时，yield 上一个 key 和它收集到的值
            yield (last_group, groups[last_group])
            assert first not in groups, f"{first} seen earlier --- violates precondition."

        groups[first].append(rest)

        last_group = first
        started = True
    
    # 返回（或可能是 yield）最后一组
    return groups


def grouper(iterable, n, fillvalue=None):
    """
    将可迭代对象 'iterable' 收集成固定长度为 'n' 的块或组。
    如果 'iterable' 的长度不是 'n' 的倍数，
    将使用 'fillvalue' 填充最后一个块。

    示例: grouper('ABCDEFG', 3, 'x') --> ('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')
    (来源: https://docs.python.org/3/library/itertools.html#itertools-recipes)

    参数:
        iterable: 任何可迭代对象。
        n (int): 块的大小。
        fillvalue (any, optional): 用于填充的值。

    返回:
        iterator: 产生元组（块）的迭代器。
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def lengths2offsets(lengths):
    """
    将一个长度（lengths）列表转换为（偏移量, 结束偏移量）对的列表。

    示例: [2, 3, 1] --> (0, 2), (2, 5), (5, 6)

    参数:
        lengths (list): 包含一系列长度的列表。

    返回:
        generator: 产生 (offset, end_offset) 元组的生成器。
    """
    offset = 0

    for length in lengths:
        yield (offset, offset + length)
        offset += length

    return


# see https://stackoverflow.com/a/45187287
class NullContextManager(object):
    """
    一个“空”的上下文管理器，它在 __enter__ 和 __exit__ 中不执行任何操作。
    当你需要根据条件“有条件地”使用 'with' 语句时，这个类很有用。
    """
    def __init__(self, dummy_resource=None):
        """
        初始化。
        参数:
            dummy_resource (any, optional): 在 __enter__ 中要返回的资源。
        """
        self.dummy_resource = dummy_resource
    
    def __enter__(self):
        """
        进入上下文时调用，返回 'dummy_resource'。
        """
        return self.dummy_resource
    
    def __exit__(self, *args):
        """
        退出上下文时调用，不执行任何操作。
        """
        pass


def load_batch_backgrounds(args, qids):
    """
    为一批查询 ID (qids) 加载“背景”上下文信息（例如相关文档）。

    它依赖于 'args' 对象中的几个属性:
    - args.qid2backgrounds: 一个从 qid 映射到 pid 列表（背景文档ID）的字典。
    - args.collection / args.collectionX: 存储实际文档内容的集合。

    参数:
        args (object): 包含配置和数据（如 qid2backgrounds）的对象。
        qids (list): 需要加载背景信息的查询 ID 列表。

    返回:
        list or None: 
            每个 qid 对应的背景文本（用 ' [SEP] ' 连接）的列表。
            如果 args.qid2backgrounds 为 None，则返回 None。
    """
    if args.qid2backgrounds is None:
        return None

    qbackgrounds = []

    for qid in qids:
        back = args.qid2backgrounds[qid] # 获取背景 pids

        if len(back) and type(back[0]) == int:
            # 假定 collection 是一个列表或数组
            x = [args.collection[pid] for pid in back]
        else:
            # 假定 collectionX 是一个字典
            x = [args.collectionX.get(pid, '') for pid in back]

        # 将所有背景文档连接成一个字符串
        x = ' [SEP] '.join(x)
        qbackgrounds.append(x)
    
    return qbackgrounds