"""setqca: native Python Qualitative Comparative Analysis.

A typed, auditable implementation of the mathematical core of crisp-set and
fuzzy-set QCA, with exact Boolean minimisation and pandas-native results.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .calibration import DirectCalibration, calibrate_crisp, calibrate_direct
from .metrics import NecessityFit, SufficiencyFit, necessity, sufficiency
from .minimize import BooleanSolution, Implicant, minimize
from .models import CSQCA, FSQCA, Direction
from .results import FittedSolution, QCAResult
from .sets import Condition, Intersection, Negation, SetExpression, Union
from .truth_table import TruthCode, TruthTable, TruthTableRow, build_truth_table

try:
    __version__ = _version("setqca")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "CSQCA",
    "FSQCA",
    "BooleanSolution",
    "Condition",
    "DirectCalibration",
    "Direction",
    "FittedSolution",
    "Implicant",
    "Intersection",
    "NecessityFit",
    "Negation",
    "QCAResult",
    "SetExpression",
    "SufficiencyFit",
    "TruthCode",
    "TruthTable",
    "TruthTableRow",
    "Union",
    "__version__",
    "build_truth_table",
    "calibrate_crisp",
    "calibrate_direct",
    "minimize",
    "necessity",
    "sufficiency",
]
