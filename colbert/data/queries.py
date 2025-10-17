# 文件名: colbert/data/queries.py

import os
import ujson

# 导入 Run 类，用于管理实验的运行环境
from colbert.infra.run import Run
# 从评估加载器中导入 load_queries 函数
from colbert.evaluation.loaders import load_queries


class Queries:
    """
    表示一个查询（问题）的集合。
    此类可以从文件路径加载数据，也接受一个已经加载的字典。
    """
    def __init__(self, path=None, data=None):
        """
        初始化 Queries 对象。

        Args:
            path (str, optional): 查询文件的路径 (可以是 .tsv 或 .json 格式)。
            data (dict, optional): 一个包含查询内容的字典 {qid: query_text}。
        """
        self.path = path

        if data:
            assert isinstance(data, dict), f"data 参数必须是字典类型, 而不是 {type(data)}"
        
        # 如果 data 参数不为空则加载 data，否则从 path 加载文件
        self._load_data(data) or self._load_file(path)
    
    def __len__(self):
        """返回查询的数量。"""
        return len(self.data)

    def __iter__(self):
        """使 Queries 对象可迭代，遍历其 (qid, query_text) 对。"""
        return iter(self.data.items())

    def provenance(self):
        """返回数据来源的路径。"""
        return self.path
    
    def toDict(self):
        """将对象的来源信息转换为字典格式。"""
        return {'provenance': self.provenance()}

    def _load_data(self, data):
        """从内存中的字典加载查询数据。"""
        if data is None:
            return None

        self.data = {}
        self._qas = {}

        for qid, content in data.items():
            if isinstance(content, dict):
                # 如果 content 是字典，则假定为 QA 格式
                self.data[qid] = content['question']
                self._qas[qid] = content
            else:
                self.data[qid] = content

        if len(self._qas) == 0:
            del self._qas

        return True

    def _load_file(self, path):
        """根据文件扩展名选择合适的加载方法。"""
        if not path.endswith('.json'):
            self.data = load_queries(path)
            return True
        
        # 加载问答（QA）格式的 JSON 文件
        self.data = {}
        self._qas = {}

        with open(path) as f:
            for line in f:
                qa = ujson.loads(line)
                assert qa['qid'] not in self.data
                self.data[qa['qid']] = qa['question']
                self._qas[qa['qid']] = qa

        return self.data

    def qas(self):
        """返回完整的问答（QA）数据字典。"""
        return dict(self._qas)

    def __getitem__(self, key):
        """支持通过 qid 访问单个查询文本。"""
        return self.data[key]

    # 以下方法提供了标准的字典接口
    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def save(self, new_path):
        """将查询数据以 'qid\tquery\n' 的格式保存到新的 .tsv 文件中。"""
        assert new_path.endswith('.tsv')
        assert not os.path.exists(new_path), f"文件 {new_path} 已存在，无法覆盖。"

        with Run().open(new_path, 'w') as f:
            for qid, content in self.data.items():
                content = f'{qid}\t{content}\n'
                f.write(content)
            
            return f.name

    def save_qas(self, new_path):
        """将完整的问答（QA）数据保存到新的 .json 文件中。"""
        assert new_path.endswith('.json')
        assert not os.path.exists(new_path), f"文件 {new_path} 已存在，无法覆盖。"

        with open(new_path, 'w') as f:
            for qid, qa in self._qas.items():
                qa['qid'] = qid
                f.write(ujson.dumps(qa) + '\n')

    @classmethod
    def cast(cls, obj):
        """
        一个类方法，用于将不同类型的输入（路径字符串、字典、列表或 Queries 对象本身）
        统一转换为 Queries 对象。
        """
        if type(obj) is str:
            return cls(path=obj)
        if isinstance(obj, dict) or isinstance(obj, list):
            return cls(data=obj)
        if type(obj) is cls:
            return obj
        assert False, f"无法将类型为 {type(obj)} 的对象转换为 Queries"