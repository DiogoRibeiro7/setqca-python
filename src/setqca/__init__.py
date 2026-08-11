"""setqca: native Python Qualitative Comparative Analysis.

A typed, auditable implementation of the mathematical core of crisp-set and
fuzzy-set QCA, with exact Boolean minimisation and pandas-native results.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .calibration import DirectCalibration, calibrate_crisp, calibrate_direct
from .counterfactuals import (
    CounterfactualAnalysis,
    DirectionalExpectation,
    classify_counterfactuals,
)
from .expressions import (
    Configuration,
    ExpressionSyntaxError,
    Implication,
    evaluate_expression,
    parse_expression,
    simplify_expression,
)
from .metrics import NecessityFit, SufficiencyFit, necessity, sufficiency
from .minimize import (
    BooleanSolution,
    Implicant,
    MinimalCover,
    MinimizationResult,
    PrimeImplicant,
    PrimeImplicantChart,
    build_chart,
    minimize,
    minimize_chart,
)
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
    "Configuration",
    "CounterfactualAnalysis",
    "DirectCalibration",
    "Direction",
    "DirectionalExpectation",
    "ExpressionSyntaxError",
    "FittedSolution",
    "Implicant",
    "Implication",
    "Intersection",
    "MinimalCover",
    "MinimizationResult",
    "NecessityFit",
    "Negation",
    "PrimeImplicant",
    "PrimeImplicantChart",
    "QCAResult",
    "SetExpression",
    "SufficiencyFit",
    "TruthCode",
    "TruthTable",
    "TruthTableRow",
    "Union",
    "__version__",
    "build_chart",
    "build_truth_table",
    "calibrate_crisp",
    "calibrate_direct",
    "classify_counterfactuals",
    "evaluate_expression",
    "minimize",
    "minimize_chart",
    "necessity",
    "parse_expression",
    "simplify_expression",
    "sufficiency",
]
