"""setqca: native Python Qualitative Comparative Analysis.

A typed, auditable implementation of the mathematical core of crisp-set and
fuzzy-set QCA, with exact Boolean minimisation and pandas-native results.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .analysis import (
    CaseDiagnostic,
    CaseRole,
    NecessityAnalysis,
    NecessityCandidate,
    RobustnessAnalysis,
    RobustnessGrid,
    SolutionDiagnostics,
    TermDiagnostics,
    necessity_analysis,
    robustness_analysis,
    sufficiency_diagnostics,
)
from .calibration import (
    AnchorSuggestion,
    CalibrationDiagnostics,
    CalibrationMethod,
    CalibrationResult,
    CalibrationSpec,
    DirectCalibration,
    calibrate,
    calibrate_crisp,
    calibrate_direct,
    crisp_spec,
    diagnose_calibration,
    diagnose_frame,
    direct_spec,
    indirect_spec,
    suggest_anchors,
)
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
from .multivalue import MVQCA, MultiValueDomain, MultiValueResult, MultiValueTruthTable
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
    "MVQCA",
    "AnchorSuggestion",
    "BooleanSolution",
    "CalibrationDiagnostics",
    "CalibrationMethod",
    "CalibrationResult",
    "CalibrationSpec",
    "CaseDiagnostic",
    "CaseRole",
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
    "MultiValueDomain",
    "MultiValueResult",
    "MultiValueTruthTable",
    "NecessityAnalysis",
    "NecessityCandidate",
    "NecessityFit",
    "Negation",
    "PrimeImplicant",
    "PrimeImplicantChart",
    "QCAResult",
    "RobustnessAnalysis",
    "RobustnessGrid",
    "SetExpression",
    "SolutionDiagnostics",
    "SufficiencyFit",
    "TermDiagnostics",
    "TruthCode",
    "TruthTable",
    "TruthTableRow",
    "Union",
    "__version__",
    "build_chart",
    "build_truth_table",
    "calibrate",
    "calibrate_crisp",
    "calibrate_direct",
    "classify_counterfactuals",
    "crisp_spec",
    "diagnose_calibration",
    "diagnose_frame",
    "direct_spec",
    "evaluate_expression",
    "indirect_spec",
    "minimize",
    "minimize_chart",
    "necessity",
    "necessity_analysis",
    "parse_expression",
    "robustness_analysis",
    "simplify_expression",
    "sufficiency",
    "sufficiency_diagnostics",
    "suggest_anchors",
]
