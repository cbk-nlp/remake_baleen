# 文件名: colbert/indexing/loaders.py

import re
import os
import ujson


def get_parts(directory):
    """
    (此函数目前似乎未在核心代码中被积极使用)
    扫描一个目录，查找所有以 '.pt' 结尾并以数字命名的文件，
    返回这些文件的编号和完整路径。
    """
    extension = '.pt'
    parts = sorted([int(filename[:-len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])
    assert list(range(len(parts))) == parts, "索引部分文件不连续"

    parts_paths = [os.path.join(directory, f'{filename}{extension}') for filename in parts]
    samples_paths = [os.path.join(directory, f'{filename}.sample') for filename in parts]
    return parts, parts_paths, samples_paths


def load_doclens(directory, flatten=True):
    """
    从索引目录加载所有文档长度 (doclens) 文件，并可以选择将它们合并成一个列表。

    doclens 文件通常被命名为 `doclens.0.json`, `doclens.1.json`, ...

    Args:
        directory (str): 索引目录的路径。
        flatten (bool, optional): 如果为 True，将所有块的 doclens 合并成一个单一的列表。
                                  如果为 False，返回一个列表的列表。默认为 True。

    Returns:
        list: 包含文档长度的列表。
    """
    doclens_filenames = {}
    # 使用正则表达式匹配所有 doclens 文件
    for filename in os.listdir(directory):
        match = re.match(r"doclens\.(\d+)\.json", filename)
        if match:
            doclens_filenames[int(match.group(1))] = filename

    # 按块索引排序文件名
    doclens_paths = [os.path.join(directory, doclens_filenames[i])
                     for i in sorted(doclens_filenames.keys())]

    all_doclens = [ujson.load(open(filename)) for filename in doclens_paths]

    if not all_doclens:
        raise ValueError("无法从目录中加载任何 doclens 文件")

    if flatten:
        # 将多个块的 doclens 列表展平成一个列表
        return [length for sublist in all_doclens for length in sublist]
    
    return all_doclens


def get_deltas(directory):
    """
    (此函数目前似乎未在核心代码中被积极使用)
    扫描目录以查找所有残差文件 (`.residuals.pt`)。
    """
    extension = '.residuals.pt'
    parts = sorted([int(filename[:-len(extension)]) for filename in os.listdir(directory)
                    if filename.endswith(extension)])
    assert list(range(len(parts))) == parts, "残差文件不连续"
    
    parts_paths = [os.path.join(directory, f'{filename}{extension}') for filename in parts]
    return parts, parts_paths