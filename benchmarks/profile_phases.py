"""Phase-level benchmarks for the QCA pipeline.

Measures each stage separately so that optimisation follows evidence rather
than intuition:

1. truth-table construction
2. prime-implicant generation
3. chart construction
4. exact cover solving
5. end-to-end minimisation

Four dimensions are varied independently, because they do not cost the same:
the number of cases, the number of conditions, the number of sufficient
configurations, and the number of logical remainders.

Usage
-----
    python benchmarks/profile_phases.py                # the default sweep
    python benchmarks/profile_phases.py --max-width 9  # push further
    python benchmarks/profile_phases.py --markdown     # table for the docs

Runtime grows sharply with the number of conditions, so the defaults are
chosen to finish in seconds and are safe to run in CI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from random import Random
from time import perf_counter

import pandas as pd

from setqca import build_truth_table
from setqca.minimize import build_chart, prime_implicants
from setqca.minimize.qmc import solve_minimum_cover


@dataclass(frozen=True, slots=True)
class Timing:
    """One measured configuration."""

    label: str
    cases: int
    conditions: int
    positives: int
    remainders: int
    primes: int
    truth_table_s: float
    primes_s: float
    chart_s: float
    cover_s: float

    @property
    def total_s(self) -> float:
        """Return the summed time across the measured phases."""
        return self.truth_table_s + self.primes_s + self.chart_s + self.cover_s

    @property
    def dominant(self) -> str:
        """Return the phase taking the largest share."""
        phases = {
            "truth table": self.truth_table_s,
            "primes": self.primes_s,
            "chart": self.chart_s,
            "cover": self.cover_s,
        }
        return max(phases, key=lambda name: phases[name])


def _fuzzy_frame(rng: Random, cases: int, conditions: int) -> pd.DataFrame:
    """Build calibrated data whose outcome actually depends on the conditions.

    An outcome drawn independently of the conditions makes almost every
    observed row consistent, so the minimiser collapses to the tautology and
    the benchmark measures nothing. Here the outcome follows two overlapping
    conjunctions plus noise, which is the shape QCA data normally has.
    """
    columns = {
        f"C{index}": [round(rng.uniform(0.02, 0.98), 3) for _ in range(cases)]
        for index in range(conditions)
    }
    outcome: list[float] = []
    for case in range(cases):
        memberships = [columns[f"C{index}"][case] for index in range(conditions)]
        first = min(memberships[: max(2, conditions // 2)])
        second = min(memberships[-2:]) if conditions >= 2 else memberships[0]
        noisy = max(first, second) + rng.uniform(-0.25, 0.25)
        outcome.append(round(min(0.98, max(0.02, noisy)), 3))
    columns["Y"] = outcome
    frame = pd.DataFrame(columns)
    # Nudge anything that landed on 0.5, which the truth table rejects.
    return frame.mask(frame == 0.5, 0.51)


def measure_truth_table(rng: Random, cases: int, conditions: int) -> tuple[float, object]:
    """Time truth-table construction for a random calibrated dataset."""
    frame = _fuzzy_frame(rng, cases, conditions)
    names = [f"C{index}" for index in range(conditions)]
    start = perf_counter()
    table = build_truth_table(frame, outcome="Y", conditions=names, inclusion_cutoff=0.75)
    return perf_counter() - start, table


def measure_minimisation(
    on_set: set[int], dont_cares: set[int], width: int
) -> tuple[float, float, float, int]:
    """Time prime generation, chart construction and cover solving separately."""
    start = perf_counter()
    primes = prime_implicants(on_set, dont_cares, width)
    primes_s = perf_counter() - start

    start = perf_counter()
    chart = build_chart(on_set, dont_cares=dont_cares, width=width)
    chart_s = perf_counter() - start

    covered = [frozenset(m for m in on_set if prime.covers(m)) for prime in primes]
    literals = [prime.literals for prime in primes]
    start = perf_counter()
    solve_minimum_cover(covered, literals, on_set, max_solutions=32)
    cover_s = perf_counter() - start

    return primes_s, chart_s, cover_s, len(chart.primes)


def sweep_conditions(rng: Random, max_width: int, cases: int) -> list[Timing]:
    """Vary the number of conditions, holding the case count fixed."""
    results: list[Timing] = []
    for width in range(3, max_width + 1):
        truth_table_s, table = measure_truth_table(rng, cases, width)
        on_set = table.positive_minterms  # type: ignore[attr-defined]
        remainders = table.remainder_minterms  # type: ignore[attr-defined]
        if not on_set:
            continue
        primes_s, chart_s, cover_s, primes = measure_minimisation(on_set, remainders, width)
        results.append(
            Timing(
                label=f"conditions={width}",
                cases=cases,
                conditions=width,
                positives=len(on_set),
                remainders=len(remainders),
                primes=primes,
                truth_table_s=truth_table_s,
                primes_s=primes_s,
                chart_s=chart_s,
                cover_s=cover_s,
            )
        )
    return results


def sweep_cases(rng: Random, conditions: int, counts: tuple[int, ...]) -> list[Timing]:
    """Vary the number of cases, holding the property space fixed."""
    results: list[Timing] = []
    for cases in counts:
        truth_table_s, table = measure_truth_table(rng, cases, conditions)
        on_set = table.positive_minterms  # type: ignore[attr-defined]
        remainders = table.remainder_minterms  # type: ignore[attr-defined]
        if not on_set:
            continue
        primes_s, chart_s, cover_s, primes = measure_minimisation(on_set, remainders, conditions)
        results.append(
            Timing(
                label=f"cases={cases}",
                cases=cases,
                conditions=conditions,
                positives=len(on_set),
                remainders=len(remainders),
                primes=primes,
                truth_table_s=truth_table_s,
                primes_s=primes_s,
                chart_s=chart_s,
                cover_s=cover_s,
            )
        )
    return results


def sweep_density(rng: Random, width: int, densities: tuple[float, ...]) -> list[Timing]:
    """Vary how many configurations are sufficient, at a fixed width."""
    results: list[Timing] = []
    universe = set(range(2**width))
    for density in densities:
        on_set = {m for m in universe if rng.random() < density}
        if not on_set:
            continue
        primes_s, chart_s, cover_s, primes = measure_minimisation(on_set, set(), width)
        results.append(
            Timing(
                label=f"density={density:g}",
                cases=0,
                conditions=width,
                positives=len(on_set),
                remainders=0,
                primes=primes,
                truth_table_s=0.0,
                primes_s=primes_s,
                chart_s=chart_s,
                cover_s=cover_s,
            )
        )
    return results


def sweep_remainders(rng: Random, width: int, shares: tuple[float, ...]) -> list[Timing]:
    """Vary how many configurations are remainders, at a fixed on-set."""
    results: list[Timing] = []
    universe = set(range(2**width))
    on_set = {m for m in universe if rng.random() < 0.15}
    if not on_set:
        return results
    rest = sorted(universe - on_set)
    for share in shares:
        count = int(len(rest) * share)
        dont_cares = set(rest[:count])
        primes_s, chart_s, cover_s, primes = measure_minimisation(on_set, dont_cares, width)
        results.append(
            Timing(
                label=f"remainders={share:g}",
                cases=0,
                conditions=width,
                positives=len(on_set),
                remainders=len(dont_cares),
                primes=primes,
                truth_table_s=0.0,
                primes_s=primes_s,
                chart_s=chart_s,
                cover_s=cover_s,
            )
        )
    return results


def render(results: list[Timing], *, markdown: bool) -> str:
    """Format the timings as a plain or Markdown table."""
    header = (
        "label",
        "cases",
        "cond",
        "pos",
        "rem",
        "primes",
        "table_s",
        "primes_s",
        "chart_s",
        "cover_s",
        "total_s",
        "dominant",
    )
    rows = [
        (
            item.label,
            str(item.cases),
            str(item.conditions),
            str(item.positives),
            str(item.remainders),
            str(item.primes),
            f"{item.truth_table_s:.4f}",
            f"{item.primes_s:.4f}",
            f"{item.chart_s:.4f}",
            f"{item.cover_s:.4f}",
            f"{item.total_s:.4f}",
            item.dominant,
        )
        for item in results
    ]
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    lines = ["  ".join(name.rjust(widths[i]) for i, name in enumerate(header))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)) for row in rows)
    return "\n".join(lines)


def main() -> None:
    """Run the sweeps and print the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-width", type=int, default=8)
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    # Each sweep gets its own generator. Sharing one would make every table
    # depend on how much randomness the tables before it happened to consume,
    # so --max-width would silently change the numbers further down.
    sections = [
        ("Conditions", sweep_conditions(Random(args.seed), args.max_width, args.cases)),
        ("Cases", sweep_cases(Random(args.seed + 1), 6, (10, 25, 50, 100, 250))),
        ("Sufficient share", sweep_density(Random(args.seed + 2), 7, (0.1, 0.25, 0.5, 0.75))),
        ("Remainder share", sweep_remainders(Random(args.seed + 3), 7, (0.0, 0.25, 0.5, 1.0))),
    ]
    for title, results in sections:
        if not results:
            continue
        print(f"\n### {title}\n" if args.markdown else f"\n=== {title} ===")
        print(render(results, markdown=args.markdown))


if __name__ == "__main__":
    main()
