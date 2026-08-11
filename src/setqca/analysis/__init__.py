"""Higher-level analyses built on the set-theoretic core.

These screen and diagnose rather than compute a single quantity: they take
calibrated data and report what it supports, together with the caveats a reader
needs in order to judge the claim.

The subpackage exists partly for a mundane reason worth stating: a module named
``setqca.necessity`` would shadow the :func:`setqca.necessity` function on the
package namespace, so the analysis modules live one level down.
"""

from __future__ import annotations

from .necessity import (
    NecessityAnalysis,
    NecessityCandidate,
    necessity_analysis,
)
from .sufficiency import (
    CaseDiagnostic,
    CaseRole,
    SolutionDiagnostics,
    TermDiagnostics,
    classify_case,
    sufficiency_diagnostics,
)

__all__ = [
    "CaseDiagnostic",
    "CaseRole",
    "NecessityAnalysis",
    "NecessityCandidate",
    "SolutionDiagnostics",
    "TermDiagnostics",
    "classify_case",
    "necessity_analysis",
    "sufficiency_diagnostics",
]
