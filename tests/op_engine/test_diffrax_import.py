"""The optional adapter must leave the base numerical installation usable."""

from __future__ import annotations

import subprocess  # noqa: S404
import sys


def test_diffrax_module_import_needs_no_optional_backend_or_compiler() -> None:
    """A fresh import works with all three optional dependencies explicitly blocked."""
    script = """
import importlib.abc
import sys
from types import SimpleNamespace

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'diffrax', 'jax', 'op_system'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockOptional())
from op_engine.diffrax import compiled_diffrax_term
assert not {'diffrax', 'jax', 'op_system'}.intersection(sys.modules)
compiled = SimpleNamespace(
    state_names=('x',), param_names=(), template_shapes={'x': ()},
    eval_fn=lambda t,y,**p: y, pytree_eval_fn=lambda t,y,**p: y,
    meta={}, operators=(), history_requirements=(),
)
try:
    compiled_diffrax_term(compiled)
except ImportError as error:
    assert 'op_engine[diffrax]' in str(error)
else:
    raise AssertionError('Missing optional dependency was not rejected')
"""
    result = subprocess.run(  # noqa: S603 - trusted isolated import contract
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
