"""LSY drone racing package for the Autonomous Drone Racing class @ TUM."""

# Python 3.13 changed Generic.__class_getitem__ checks, breaking warp 1.6.2's
# `wp.array[int]` annotation syntax inside mujoco_warp @wp.struct decorators.
# We intercept mujoco.mjx.warp at import time and broaden its except clause so
# TypeError is treated as "warp backend unavailable" — the JAX path still works.
def _patch_mujoco_warp_py313() -> None:
    import sys
    import importlib.abc

    _TARGET = "mujoco.mjx.warp"
    if _TARGET in sys.modules:
        return  # already imported, nothing to do

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            # Skip ALL instances of our finder to prevent recursion between
            # multiple installed copies of _Finder (e.g. from __init__ and
            # from train_noguide_jax.py both installing one).
            for finder in sys.meta_path:
                if isinstance(finder, _Finder):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and getattr(spec, "origin", None):
                    return _PatchLoader.wrap(spec)
            return None

    class _PatchLoader(importlib.abc.Loader):
        def __init__(self, orig, origin):
            self._orig = orig
            self._origin = origin

        @staticmethod
        def wrap(spec):
            spec.loader = _PatchLoader(spec.loader, spec.origin)
            return spec

        def create_module(self, spec):
            m = getattr(self._orig, "create_module", lambda s: None)(spec)
            return m

        def exec_module(self, module):
            with open(self._origin) as f:
                src = f.read()
            src = src.replace(
                "except (ImportError, RuntimeError) as e:",
                "except (ImportError, RuntimeError, TypeError, AttributeError) as e:",
            )
            exec(compile(src, self._origin, "exec"), module.__dict__)

    sys.meta_path.insert(0, _Finder())

_patch_mujoco_warp_py313()

from crazyflow.utils import enable_cache

import lsy_drone_racing.envs  # noqa: F401, register environments with gymnasium

enable_cache()  # Enable persistent caching of jax functions
