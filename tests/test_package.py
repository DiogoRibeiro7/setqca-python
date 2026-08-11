"""Package-level contract tests."""

from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

import pytest

import setqca

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_installed_distribution_metadata() -> None:
    assert setqca.__version__ == version("setqca")


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert setqca.__version__ == pyproject["project"]["version"]


def test_citation_metadata_tracks_the_package_version() -> None:
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f'version: "{setqca.__version__}"' in citation


def test_every_advertised_name_is_importable() -> None:
    assert len(set(setqca.__all__)) == len(setqca.__all__), "__all__ contains duplicates"
    for name in setqca.__all__:
        assert hasattr(setqca, name), f"{name} is advertised in __all__ but not importable"


def test_public_api_covers_the_documented_entry_points() -> None:
    expected = {
        "CSQCA",
        "FSQCA",
        "Condition",
        "build_truth_table",
        "calibrate_direct",
        "minimize",
        "necessity",
        "sufficiency",
    }
    assert expected <= set(setqca.__all__)


@pytest.mark.parametrize(
    "name",
    ["necessity", "sufficiency", "minimize", "calibrate_direct", "evaluate_expression"],
)
def test_exported_callables_are_not_shadowed_by_submodules(name: str) -> None:
    """A submodule sharing a name with an export would silently replace it.

    Importing ``setqca.necessity`` as a module, for instance, rebinds the
    attribute on the package and turns ``from setqca import necessity`` into a
    module import. Analysis modules therefore live under ``setqca.analysis``.
    """
    exported = getattr(setqca, name)
    assert callable(exported), f"setqca.{name} is {type(exported).__name__}, not a callable"
    assert not isinstance(exported, ModuleType)


def test_package_ships_a_py_typed_marker() -> None:
    assert (Path(setqca.__file__).parent / "py.typed").is_file()
