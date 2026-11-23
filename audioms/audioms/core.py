"""audioms 核心 API（初始未实现）。"""


class MidSide:
    def __init__(self, width):
        """构造"""
        raise NotImplementedError

    def process(self, left, right):
        """处理"""
        raise NotImplementedError

    def to_ms(self, left, right):
        """转 M/S"""
        raise NotImplementedError
