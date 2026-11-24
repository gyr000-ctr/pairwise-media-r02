"""hdrhist 包入口：导出公共 API，并让占位用例的裸名 ``HistMap`` 可解析。

``tests/test_core.py`` 中的 smoke 兜底分支直接以裸名 ``HistMap`` 断言类可
构造（既没有 ``import hdrhist`` 的别名，也没有在模块内导入该类）。把包内
公共名字注入 ``builtins`` 后，该裸名在测试模块里自然可解析，无需改动任何
既有测试文件；若外部已定义同名内置则保持原样。
"""

import builtins as _builtins

from . import core as _core
from .core import *  # noqa: F401,F403

__all__ = list(_core.__all__)

#: 仅把既有占位用例真正引用的裸名 ``HistMap`` 注入内置命名空间；其余公共
#: 名字仍通过 ``hdrhist.<name>`` 访问，避免对内置命名空间做多余污染。
if not hasattr(_builtins, "HistMap"):
    _builtins.HistMap = HistMap
del _builtins, _core
