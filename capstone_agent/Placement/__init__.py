import sys as _sys
import os as _os

_pkg_dir = _os.path.dirname(__file__)
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

_parent_dir = _os.path.normpath(_os.path.join(_pkg_dir, ".."))
if _parent_dir not in _sys.path:
    _sys.path.insert(0, _parent_dir)
