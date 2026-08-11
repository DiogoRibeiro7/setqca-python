"""Parity tests against the reference R ``QCA`` implementation.

Golden values live in ``validation/fixtures/r_qca.json`` and are committed, so
these tests run everywhere without R installed. Regenerate them deliberately
with::

    Rscript validation/r/generate_fixtures.R validation/fixtures/r_qca.json

A change in that fixture is a change in what the reference implementation says,
and should be reviewed as carefully as a change to the source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from setqca import (
    FSQCA,
    Condition,
    QCAResult,
    SetExpression,
    build_truth_table,
    calibrate_direct,
    necessity,
    necessity_analysis,
    sufficiency,
    sufficiency_diagnostics,
)

pytestmark = pytest.mark.parity

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "validation" / "fixtures" / "r_qca.json"

if not FIXTURE_PATH.is_file():  # pragma: no cover - only when fixtures are absent
    pytest.skip(f"missing R parity fixture: {FIXTURE_PATH}", allow_module_level=True)

FIXTURE: dict[str, Any] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

# R's logistic calibration and the truth-table statistics are reported to full
# double precision, so parity is expected to hold far tighter than any
# substantive threshold.
TOLERANCE = 1e-9


def _ids(records: list[dict[str, Any]]) -> list[str]:
    return [record["id"] for record in records]


def _as_list(value: str | list[str]) -> list[str]:
    """Normalise a jsonlite value that may have been auto-unboxed to a scalar."""
    if isinstance(value, str):
        return [value]
    return list(value)


def _as_str_list(value: str | list[str] | None) -> list[str]:
    """Normalise a jsonlite string field that may have been auto-unboxed."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _as_int_list(value: int | list[int] | None) -> list[int]:
    """Normalise a jsonlite numeric field that may have been auto-unboxed."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return list(value)


def _parse_expression(text: str) -> SetExpression:
    """Parse R ``QCA`` notation such as ``DEV*~URB`` or ``DEV+URB``."""
    terms = text.split("+")
    parsed: list[SetExpression] = []
    for term in terms:
        literals: list[SetExpression] = []
        for raw in term.split("*"):
            token = raw.strip()
            negated = token.startswith("~")
            condition: SetExpression = Condition(token.lstrip("~"))
            literals.append(~condition if negated else condition)
        current = literals[0]
        for literal in literals[1:]:
            current = current & literal
        parsed.append(current)
    combined = parsed[0]
    for item in parsed[1:]:
        combined = combined | item
    return combined


def _canonical_solution(implicants: list[str]) -> frozenset[frozenset[str]]:
    """Reduce a solution to comparable literal sets, ignoring notation and order."""
    return frozenset(
        frozenset(literal.strip() for literal in implicant.split("*")) for implicant in implicants
    )


def _canonical_solutions(
    solutions: list[str | list[str]],
) -> set[frozenset[frozenset[str]]]:
    return {_canonical_solution(_as_list(solution)) for solution in solutions}


def _frame(analysis: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(analysis["data"])
    frame.index = pd.Index(analysis["case_ids"])
    return frame


def _has_crossover(frame: pd.DataFrame, columns: list[str]) -> bool:
    values = frame[columns].to_numpy(dtype=float)
    return bool(np.isclose(values, 0.5, atol=1e-12).any())


def _fit(analysis: dict[str, Any]) -> QCAResult:
    frame = _frame(analysis)
    conditions = list(analysis["conditions"])
    model = FSQCA(consistency=float(analysis["incl_cut"]), frequency=int(analysis["n_cut"]))
    return model.fit(frame, outcome=analysis["outcome"], conditions=conditions)


# ---------------------------------------------------------------------------
# Direct calibration
# ---------------------------------------------------------------------------


SNAP_LOW = float(FIXTURE["r_snapping"]["low"])
SNAP_HIGH = float(FIXTURE["r_snapping"]["high"])


def _calibrate(case: dict[str, Any]) -> npt.NDArray[np.float64]:
    full_out, crossover, full_in = case["thresholds"]
    return calibrate_direct(
        case["values"],
        full_out=full_out,
        crossover=crossover,
        full_in=full_in,
        idm=case["idm"],
        logistic=case["logistic"],
    )


@pytest.mark.parametrize("case", FIXTURE["calibration"], ids=_ids(FIXTURE["calibration"]))
def test_direct_calibration_matches_r(case: dict[str, Any]) -> None:
    """Calibration agrees with R wherever R does not snap the result.

    ``QCA::calibrate`` ends with ``fs[fs < 1e-04] <- 0; fs[fs > 0.9999] <- 1``,
    so memberships in those tails come back as exactly 0 or 1 rather than as the
    value of the transformation. setqca reports the transformation itself. Every
    other point must agree to full double precision.
    """
    obtained = _calibrate(case)
    divergences: list[str] = []

    for value, expected, got in zip(case["values"], case["expected"], obtained, strict=True):
        if abs(expected - got) <= TOLERANCE:
            continue
        snapped_low = expected == 0.0 and got < SNAP_LOW
        snapped_high = expected == 1.0 and got > SNAP_HIGH
        if snapped_low or snapped_high:
            # R's snapping rule applies here, and setqca agrees it would.
            continue
        divergences.append(f"x={value}: R={expected!r} setqca={got!r}")

    assert not divergences, "unexplained divergence from R:\n  " + "\n  ".join(divergences)


def test_r_snapping_is_the_only_calibration_divergence() -> None:
    """Pin the known divergence so it cannot silently widen.

    If R ever stops snapping, or setqca starts, this test fails and the
    documented divergence has to be revisited.
    """
    snapped: list[tuple[float, float]] = []
    for case in FIXTURE["calibration"]:
        for value, expected, got in zip(
            case["values"], case["expected"], _calibrate(case), strict=True
        ):
            if abs(expected - got) > TOLERANCE:
                snapped.append((value, got))

    assert snapped, "expected at least one snapped extreme in the fixture"
    for value, got in snapped:
        assert got < SNAP_LOW or got > SNAP_HIGH, (
            f"x={value} diverged at {got!r}, which is outside R's snapping tails; "
            "this is a real parity failure, not the documented rounding rule"
        )


# ---------------------------------------------------------------------------
# Parameters of fit
# ---------------------------------------------------------------------------

_SUFFICIENCY = [c for c in FIXTURE["pof"] if c["relation"] == "sufficiency"]
_NECESSITY = [c for c in FIXTURE["pof"] if c["relation"] == "necessity"]


@pytest.mark.parametrize("case", _SUFFICIENCY, ids=[c["expression"] for c in _SUFFICIENCY])
def test_sufficiency_matches_r(case: dict[str, Any]) -> None:
    frame = _frame(FIXTURE["analyses"][0])
    membership = _parse_expression(case["expression"]).evaluate(frame)
    fit = sufficiency(membership, frame[case["outcome"]].to_numpy())

    assert fit.consistency == pytest.approx(case["consistency"], abs=TOLERANCE)
    assert fit.coverage == pytest.approx(case["coverage"], abs=TOLERANCE)
    if case.get("pri") is not None:
        assert fit.pri == pytest.approx(case["pri"], abs=TOLERANCE)


@pytest.mark.parametrize("case", _NECESSITY, ids=[c["expression"] for c in _NECESSITY])
def test_necessity_matches_r(case: dict[str, Any]) -> None:
    frame = _frame(FIXTURE["analyses"][0])
    membership = _parse_expression(case["expression"]).evaluate(frame)
    fit = necessity(membership, frame[case["outcome"]].to_numpy())

    assert fit.consistency == pytest.approx(case["consistency"], abs=TOLERANCE)
    assert fit.coverage == pytest.approx(case["coverage"], abs=TOLERANCE)
    if case.get("ron") is not None:
        assert fit.ron == pytest.approx(case["ron"], abs=TOLERANCE)


# ---------------------------------------------------------------------------
# Truth tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("analysis", FIXTURE["analyses"], ids=_ids(FIXTURE["analyses"]))
def test_truth_table_matches_r(analysis: dict[str, Any]) -> None:
    frame = _frame(analysis)
    conditions = list(analysis["conditions"])
    table = build_truth_table(
        frame,
        outcome=analysis["outcome"],
        conditions=conditions,
        inclusion_cutoff=float(analysis["incl_cut"]),
        frequency_cutoff=int(analysis["n_cut"]),
        allow_crossover_cases=_has_crossover(frame, conditions),
    )
    obtained = {row.minterm: row for row in table.rows}

    assert len(obtained) == len(analysis["truth_table"])

    for expected in analysis["truth_table"]:
        row = obtained[expected["minterm"]]
        assert tuple(expected["configuration"]) == row.configuration
        assert row.frequency == expected["n"], f"case count differs at {expected['minterm']}"
        assert row.outcome == expected["out"], f"row code differs at {expected['minterm']}"
        if expected["consistency"] is not None:
            assert row.consistency == pytest.approx(expected["consistency"], abs=TOLERANCE), (
                f"consistency differs at minterm {expected['minterm']}"
            )
        if expected["pri"] is not None:
            assert row.pri == pytest.approx(expected["pri"], abs=TOLERANCE), (
                f"PRI differs at minterm {expected['minterm']}"
            )


# ---------------------------------------------------------------------------
# Minimisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("analysis", FIXTURE["analyses"], ids=_ids(FIXTURE["analyses"]))
def test_conservative_solution_matches_r(analysis: dict[str, Any]) -> None:
    if not analysis["conservative"]:
        pytest.skip("R produced no conservative solution for this analysis")
    result = _fit(analysis)
    obtained = {
        _canonical_solution(solution.expression(result.conditions).split(" + "))
        for solution in result.conservative
    }
    assert obtained == _canonical_solutions(analysis["conservative"])


@pytest.mark.parametrize("analysis", FIXTURE["analyses"], ids=_ids(FIXTURE["analyses"]))
def test_parsimonious_solution_matches_r(analysis: dict[str, Any]) -> None:
    if not analysis["parsimonious"]:
        pytest.skip("R produced no parsimonious solution for this analysis")
    result = _fit(analysis)
    obtained = {
        _canonical_solution(solution.expression(result.conditions).split(" + "))
        for solution in result.parsimonious
    }
    assert obtained == _canonical_solutions(analysis["parsimonious"])


# ---------------------------------------------------------------------------
# Per-term fit, including unique coverage
# ---------------------------------------------------------------------------

_TERM_FITS = [
    (analysis, record)
    for analysis in FIXTURE["analyses"]
    for record in analysis.get("term_fits", [])
]


@pytest.mark.parametrize(
    ("analysis", "record"),
    _TERM_FITS,
    ids=[f"{a['id']}-{r['family']}-{r['term']}" for a, r in _TERM_FITS],
)
def test_term_fit_matches_r(analysis: dict[str, Any], record: dict[str, Any]) -> None:
    """Per-term consistency, PRI, raw coverage and unique coverage."""
    frame = _frame(analysis)
    family_terms = [
        item["term"] for item in analysis["term_fits"] if item["family"] == record["family"]
    ]
    diagnostics = sufficiency_diagnostics(frame, outcome=analysis["outcome"], terms=family_terms)
    term = next(item for item in diagnostics.terms if item.expression == record["term"])

    assert term.fit.consistency == pytest.approx(record["consistency"], abs=TOLERANCE)
    assert term.fit.pri == pytest.approx(record["pri"], abs=TOLERANCE)
    assert term.fit.coverage == pytest.approx(record["raw_coverage"], abs=TOLERANCE)

    if record.get("unique_coverage") is None:
        # R leaves covU undefined for a one-term solution. setqca reports the
        # raw coverage instead: with nothing to share with, everything the term
        # covers is uniquely covered by it.
        assert len(family_terms) == 1
        assert term.unique_coverage == pytest.approx(term.fit.coverage)
    else:
        assert term.unique_coverage == pytest.approx(record["unique_coverage"], abs=TOLERANCE)


@pytest.mark.parametrize(
    ("analysis", "record"),
    _TERM_FITS,
    ids=[f"{a['id']}-{r['family']}-{r['term']}" for a, r in _TERM_FITS],
)
def test_cases_in_a_term_match_r(analysis: dict[str, Any], record: dict[str, Any]) -> None:
    """R's ``cases`` column lists membership in the term, above the crossover.

    The case typology splits those further into typical and deviant-in-degree,
    so the union of those two roles is what corresponds to R's list.
    """
    frame = _frame(analysis)
    family_terms = [
        item["term"] for item in analysis["term_fits"] if item["family"] == record["family"]
    ]
    diagnostics = sufficiency_diagnostics(frame, outcome=analysis["outcome"], terms=family_terms)
    term = next(item for item in diagnostics.terms if item.expression == record["term"])

    in_term = {item.case for item in term.cases if item.term_membership > 0.5}
    assert in_term == set(_as_str_list(record["cases"]))
    assert term.frequency == len(in_term)
    assert set(term.typical) <= in_term


# ---------------------------------------------------------------------------
# Systematic necessity screening
# ---------------------------------------------------------------------------

_SCREENS = FIXTURE.get("necessity_screens", [])


@pytest.mark.parametrize("screen", _SCREENS, ids=_ids(_SCREENS))
def test_necessity_screening_matches_r(screen: dict[str, Any]) -> None:
    """Every candidate, presence and absence alike, must agree with R."""
    frame = _frame(screen)
    analysis = necessity_analysis(
        frame,
        outcome=screen["outcome"],
        conditions=list(screen["conditions"]),
        max_disjunction_size=2,
    )
    obtained = {item.expression: item.fit for item in analysis.candidates}

    for expected in screen["candidates"]:
        expression = expected["expression"]
        if expression not in obtained:
            # R's extra unions are checked through the expression evaluator.
            membership = _parse_expression(expression).evaluate(frame)
            fit = necessity(membership, frame[screen["outcome"]].to_numpy())
        else:
            fit = obtained[expression]

        assert fit.consistency == pytest.approx(expected["consistency"], abs=TOLERANCE), (
            f"consistency differs for {expression}"
        )
        assert fit.coverage == pytest.approx(expected["coverage"], abs=TOLERANCE), (
            f"coverage differs for {expression}"
        )
        assert fit.ron == pytest.approx(expected["ron"], abs=TOLERANCE), (
            f"RoN differs for {expression}"
        )


@pytest.mark.parametrize("screen", _SCREENS, ids=_ids(_SCREENS))
def test_trivial_necessity_is_flagged_where_relevance_is_low(screen: dict[str, Any]) -> None:
    """LIT+STB reaches consistency 0.995 in R but RoN 0.38 — necessary-looking, trivial."""
    frame = _frame(screen)
    analysis = necessity_analysis(
        frame,
        outcome=screen["outcome"],
        conditions=list(screen["conditions"]),
        max_disjunction_size=2,
    )
    union = next(item for item in analysis.candidates if item.expression == "LIT+STB")
    assert union.fit.consistency > 0.99
    assert union.fit.ron < 0.5
    assert union.trivial is True
    assert union.necessary is False


# ---------------------------------------------------------------------------
# Intermediate solutions and counterfactuals
# ---------------------------------------------------------------------------

_INTERMEDIATE = FIXTURE.get("intermediate", [])


def _fit_intermediate(case: dict[str, Any]) -> QCAResult:
    frame = _frame(case)
    model = FSQCA(
        consistency=float(case["incl_cut"]),
        directional_expectations={name: int(value) for name, value in case["expectations"].items()},
    )
    return model.fit(frame, outcome=case["outcome"], conditions=list(case["conditions"]))


@pytest.mark.parametrize("case", _INTERMEDIATE, ids=_ids(_INTERMEDIATE))
def test_intermediate_solution_matches_r(case: dict[str, Any]) -> None:
    result = _fit_intermediate(case)
    assert result.intermediate is not None
    obtained = {
        _canonical_solution(solution.expression(result.conditions).split(" + "))
        for solution in result.intermediate
    }
    assert obtained == _canonical_solutions(case["intermediate"])


@pytest.mark.parametrize("case", _INTERMEDIATE, ids=_ids(_INTERMEDIATE))
def test_simplifying_assumptions_match_r(case: dict[str, Any]) -> None:
    analysis = _fit_intermediate(case).counterfactuals
    assert analysis is not None
    assert sorted(analysis.simplifying_assumptions) == _as_int_list(case["simplifying_assumptions"])


@pytest.mark.parametrize("case", _INTERMEDIATE, ids=_ids(_INTERMEDIATE))
def test_easy_and_difficult_counterfactuals_match_r(case: dict[str, Any]) -> None:
    analysis = _fit_intermediate(case).counterfactuals
    assert analysis is not None
    assert sorted(analysis.easy) == _as_int_list(case["easy"])
    assert sorted(analysis.difficult) == _as_int_list(case["difficult"])
