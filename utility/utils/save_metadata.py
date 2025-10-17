# 文件名: utility/utils/save_metadata.py

import os
import sys
import git
import time
import copy
import ujson
import socket

from colbert.utils.utils import dotdict


def get_metadata_only():
    """
    仅收集与运行环境相关的元数据，不包含任何命令行参数。
    """
    args = dotdict()
    repo = git.Repo(search_parent_directories=True)
    
    args.hostname = socket.gethostname()
    args.git_branch = repo.active_branch.name
    args.git_hash = repo.head.object.hexsha
    args.git_commit_datetime = str(repo.head.object.committed_datetime)
    args.current_datetime = time.strftime('%b %d, %Y ; %l:%M%p %Z (%z)')
    args.cmd = ' '.join(sys.argv)
    
    return args


def get_metadata(args):
    """
    收集完整的元数据，包括运行环境信息和传入的所有命令行参数。

    Args:
        args (Namespace or dotdict): 从 ArgumentParser 解析得到的参数对象。
    
    Returns:
        dict: 包含所有元数据和参数的字典。
    """
    # 创建一个深拷贝，以避免修改原始参数对象
    args = copy.deepcopy(args)

    # 收集环境元数据
    env_meta = get_metadata_only()
    args.update(env_meta)

    # 将嵌套的 input_arguments 对象也转换为字典，以便序列化
    if hasattr(args, 'input_arguments'):
        args.input_arguments = copy.deepcopy(args.input_arguments.__dict__)

    return dict(args)


def format_metadata(metadata):
    """将元数据字典格式化为易于阅读的 JSON 字符串。"""
    assert isinstance(metadata, dict)
    return ujson.dumps(metadata, indent=4)


def save_metadata(path, args):
    """
    将完整的元数据保存到指定的路径。

    Args:
        path (str): 保存元数据的文件路径。
        args (Namespace or dotdict): 参数对象。

    Returns:
        dict: 已保存的元数据字典。
    """
    assert not os.path.exists(path), f"元数据文件 {path} 已存在！"

    with open(path, 'w') as f:
        data = get_metadata(args)
        f.write(format_metadata(data) + '\n')

    return data