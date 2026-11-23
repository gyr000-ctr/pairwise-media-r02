import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# 测试模块以 `import polybool` 方式引用，占位用例的兜底断言直接使用裸名
# `Poly`；将包内公共名字注入内置命名空间，使其在测试模块中可解析。
import builtins

import polybool as _polybool

for _name in getattr(_polybool, "__all__", None) or dir(_polybool):
    if not _name.startswith("_"):
        setattr(builtins, _name, getattr(_polybool, _name))
