"""Regression tests for main.py macros.

These catch breakage when Ludwig's schema API changes (e.g. get_class_schema
removed in pydantic-v2 migration) before it reaches the CI docs build.
"""

import sys
import pytest

# Ludwig's schema metaclass is incompatible with pydantic 2.13 on Python 3.14+.
# CI uses Python 3.12 where everything works. Skip import-heavy tests locally
# when running on an unsupported version so CI remains the authoritative gate.
_ludwig_importable = True
try:
    import ludwig.schema.combiners  # noqa: F401
except Exception:
    _ludwig_importable = False

requires_ludwig = pytest.mark.skipif(
    not _ludwig_importable,
    reason="Ludwig schema metaclass incompatible with this Python/pydantic version (CI uses Python 3.12)",
)


def _make_env():
    """Return a minimal env stub that records registered macros."""
    macros = {}

    class Env:
        def macro(self, fn):
            macros[fn.__name__] = fn
            return fn

    env = Env()
    env._macros = macros
    return env


@requires_ludwig
def test_define_env_registers_macros():
    from main import define_env
    env = _make_env()
    define_env(env)
    for name in ("schema_class_to_yaml", "schema_class_to_fields", "schema_class_long_description"):
        assert name in env._macros, f"macro {name!r} not registered"


@requires_ludwig
def test_schema_class_to_yaml():
    from main import define_env
    from ludwig.schema.preprocessing import PreprocessingConfig
    env = _make_env()
    define_env(env)
    result = env._macros["schema_class_to_yaml"](PreprocessingConfig)
    assert isinstance(result, str)
    assert len(result) > 0


@requires_ludwig
def test_schema_class_to_fields():
    from main import define_env
    from ludwig.schema.preprocessing import PreprocessingConfig
    env = _make_env()
    define_env(env)
    result = env._macros["schema_class_to_fields"](PreprocessingConfig)
    assert isinstance(result, dict)
    assert len(result) > 0


@requires_ludwig
def test_schema_class_long_description():
    from main import define_env
    from ludwig.schema.combiners.utils import get_combiner_registry
    env = _make_env()
    define_env(env)
    # Use a combiner that has a 'type' field
    cls = get_combiner_registry()["concat"]
    result = env._macros["schema_class_long_description"](cls)
    assert isinstance(result, str)


def test_no_get_class_schema_calls():
    """Ensure main.py never calls get_class_schema (removed in pydantic v2)."""
    import ast, pathlib
    src = pathlib.Path(__file__).parent.parent / "main.py"
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "get_class_schema":
            pytest.fail(f"main.py calls get_class_schema at line {node.lineno} — use model_fields instead")
