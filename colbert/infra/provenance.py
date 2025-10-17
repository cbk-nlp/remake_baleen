# 文件名: colbert/infra/provenance.py

import inspect

class Provenance:
    """
    一个用于记录对象“来源”或“出处”的类。

    当一个 Provenance 对象被创建时，它会捕获当前的函数调用堆栈。
    这对于追踪一个数据对象（例如一个排序结果 `Ranking`）是如何通过
    一系列代码调用生成的非常有用，有助于保证实验的可复现性和调试。
    """
    def __init__(self) -> None:
        """初始化时，记录初始的调用堆栈。"""
        self.initial_stacktrace = self.stacktrace()

    def stacktrace(self):
        """
        获取并格式化当前的调用堆栈。

        Returns:
            list[str]: 一个字符串列表，每一项代表调用堆栈中的一帧。
        """
        trace = inspect.stack()
        output = []
        # 忽略堆栈顶部（当前文件内的调用）和底部（系统调用）的帧
        for frame in trace[2:-1]:
            # 格式化输出为: "文件名:行号:函数名: 代码行内容"
            frame_info = f'{frame.filename}:{frame.lineno}:{frame.function}:   {frame.code_context[0].strip()}'
            output.append(frame_info)
        return output

    def toDict(self):
        """
        将 Provenance 对象转换为字典，以便于序列化（例如保存为 JSON）。
        在序列化时，会额外记录当前的调用堆栈。
        """
        self.serialization_stacktrace = self.stacktrace()
        return dict(self.__dict__)