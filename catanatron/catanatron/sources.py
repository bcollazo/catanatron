"""Turning a ``--bot``/``--accumulator`` SOURCE into a class.

Three forms, each optionally naming a class after a ``#``::

    ./bots/mybot.py           a file, anywhere on disk
    ./bots/mybot.py#MyBot     ...naming which class in it
    mypkg.bots#MyBot          an importable module
    exec:./mybot              any program, spoken to over stdin/stdout

``#`` rather than ``:`` separates the class, because a Windows path
(``C:\\bots\\mybot.py``) already contains a colon.
"""

import importlib
import importlib.util
import inspect
import os

#: Runs the bot as a separate program.
EXEC_PREFIX = "exec:"

#: Recognized only so the error names the transport that is still missing.
UNSUPPORTED_PREFIXES = ("http://", "https://", "url:")


class SourceError(ValueError):
    """A source that cannot be resolved."""


def load_module(target: str, base_dir=None):
    """Import a file path or a module name."""
    if target.endswith(".py") or os.sep in target or "/" in target:
        path = os.path.abspath(os.path.join(base_dir or os.getcwd(), target))
        if not os.path.isfile(path):
            raise SourceError(f"no such file: {path}")
        name = "catanatron_source_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise SourceError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module(target)
    except ImportError as error:
        raise SourceError(f"cannot import {target!r}: {error}")


def load_class(source: str, base_class, base_dir=None):
    """Resolve ``source`` to a subclass of ``base_class``."""
    if source.startswith(UNSUPPORTED_PREFIXES):
        raise SourceError(
            f"{source!r}: bots over HTTP are not supported yet. Use a Python "
            f"class (./mybot.py) or any program (exec:./mybot)"
        )

    target, _, class_name = source.partition("#")
    if not target:
        raise SourceError(f"invalid source: {source!r}")
    module = load_module(target, base_dir)

    if class_name:
        found = getattr(module, class_name, None)
        if found is None:
            raise SourceError(f"{source}: {class_name!r} is not defined there")
    else:
        candidates = [
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and issubclass(obj, base_class)
            and obj.__module__ == module.__name__
        ]
        if not candidates:
            raise SourceError(
                f"{source}: no {base_class.__name__} subclass defined there"
            )
        if len(candidates) > 1:
            names = sorted(c.__name__ for c in candidates)
            raise SourceError(
                f"{source}: defines several ({', '.join(names)}); name one, "
                f"e.g. {target}#{names[0]}"
            )
        found = candidates[0]

    if not (inspect.isclass(found) and issubclass(found, base_class)):
        raise SourceError(f"{source}: {found!r} is not a {base_class.__name__}")
    return found


def identity(cls):
    """Where a class came from, stable across re-imports of the same file."""
    return f"{cls.__module__}.{cls.__qualname__}"
