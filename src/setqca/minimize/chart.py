"""The prime-implicant chart, made inspectable.

Minimisation normally reports only its answer. For a configurational analysis
the reasoning matters as much as the result: which configurations forced a term
into the solution, which terms were interchangeable, and which were never
candidates at all. This module exposes that structure.

The chart is the classical one: a row for every truth-table configuration that
must be covered, a column for every prime implicant, and a mark where the
implicant covers the configuration.

- A prime is **essential** when some configuration is covered by it alone. Every
  minimal cover contains every essential prime.
- A prime is **dominated** when another prime covers at least as much for no
  greater cost and is strictly better on one of the two. A dominated prime
  appears in no minimum cover at all.
- Primes that merely tie — same rows, same cost — are **interchangeable** rather
  than dominated. Each heads a distinct minimum cover, and all of them are
  reported.
- A configuration is **uncoverable** when no prime covers it, which means the
  chart cannot be solved at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .implicant import Implicant
from .qmc import BooleanSolution, exact_minimum_covers, prime_implicants

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import pandas as pd


@dataclass(frozen=True, slots=True)
class PrimeImplicant:
    """A prime implicant together with its position in the chart."""

    index: int
    implicant: Implicant
    covered: frozenset[int]

    @property
    def literals(self) -> int:
        """Return the number of fixed literals."""
        return self.implicant.literals

    def covers(self, minterm: int) -> bool:
        """Return whether this prime covers a configuration."""
        return self.implicant.covers(minterm)

    def as_expression(self, conditions: tuple[str, ...]) -> str:
        """Render in standard QCA notation."""
        return self.implicant.as_expression(conditions)


@dataclass(frozen=True, slots=True)
class PrimeImplicantChart:
    """The relationship between configurations and the primes that cover them."""

    on_set: frozenset[int]
    primes: tuple[PrimeImplicant, ...]

    @property
    def essential(self) -> tuple[PrimeImplicant, ...]:
        """Return the primes that every minimal cover must contain."""
        forced = {
            candidates[0]
            for minterm in sorted(self.on_set)
            if len(candidates := self.candidates_for(minterm)) == 1
        }
        return tuple(sorted(forced, key=lambda prime: prime.index))

    @property
    def dominated(self) -> tuple[PrimeImplicant, ...]:
        """Return primes that cannot appear in any minimum cover.

        A prime is dominated when another prime covers at least as much for no
        greater literal cost, and is *strictly* better on one of the two — it
        covers strictly more, or costs strictly less. These explain why a term
        that looks plausible never appears in any solution.

        Domination is deliberately strict. Two primes covering the same rows at
        the same cost are **interchangeable**, not dominated: each heads a
        distinct minimum cover, and this package reports all of them. Calling
        one of a tied pair dominated would be a tie-break, which is sound only
        when a single solution is wanted.
        """
        redundant: list[PrimeImplicant] = []
        for prime in self.primes:
            for other in self.primes:
                if other.index == prime.index:
                    continue
                at_least_as_good = prime.covered <= other.covered and other.literals <= (
                    prime.literals
                )
                strictly_better = prime.covered < other.covered or other.literals < prime.literals
                if at_least_as_good and strictly_better:
                    redundant.append(prime)
                    break
        return tuple(redundant)

    @property
    def uncoverable(self) -> frozenset[int]:
        """Return configurations no prime covers."""
        return frozenset(minterm for minterm in self.on_set if not self.candidates_for(minterm))

    def candidates_for(self, minterm: int) -> tuple[PrimeImplicant, ...]:
        """Return the primes covering one configuration, in chart order."""
        return tuple(prime for prime in self.primes if prime.covers(minterm))

    def explain(self, minterm: int, conditions: tuple[str, ...]) -> str:
        """Explain in words how one configuration can be covered.

        Parameters
        ----------
        minterm : int
            The configuration to explain.
        conditions : tuple of str
            Condition names, for rendering the implicants.
        """
        if minterm not in self.on_set:
            return f"Row {minterm} is not required to be covered."
        candidates = self.candidates_for(minterm)
        if not candidates:
            return f"Row {minterm} is covered by no prime implicant; the chart cannot be solved."
        rendered = ", ".join(prime.as_expression(conditions) for prime in candidates)
        if len(candidates) == 1:
            return (
                f"Row {minterm} is covered only by {rendered}, "
                "which is therefore essential and appears in every solution."
            )
        return f"Row {minterm} can be covered by any of: {rendered}."

    def to_frame(self) -> pd.DataFrame:
        """Return the chart as a boolean table, rows by configuration.

        Returns
        -------
        pandas.DataFrame
            Index is the configuration minterm; one column per prime, named by
            its chart index; values mark coverage.
        """
        import pandas as pd

        return pd.DataFrame(
            {
                f"P{prime.index}": [prime.covers(minterm) for minterm in sorted(self.on_set)]
                for prime in self.primes
            },
            index=pd.Index(sorted(self.on_set), name="minterm"),
        )


@dataclass(frozen=True, slots=True)
class MinimalCover:
    """One exact minimum cover, with the reason each prime is in it."""

    primes: tuple[PrimeImplicant, ...]
    essential_indices: frozenset[int]

    @property
    def size(self) -> int:
        """Return the number of implicants."""
        return len(self.primes)

    @property
    def literal_count(self) -> int:
        """Return the total number of literals."""
        return sum(prime.literals for prime in self.primes)

    @property
    def cost(self) -> tuple[int, int]:
        """Return the lexicographic cost: implicant count, then literal count."""
        return (self.size, self.literal_count)

    def as_expression(self, conditions: tuple[str, ...]) -> str:
        """Render the cover in standard QCA notation."""
        return " + ".join(prime.as_expression(conditions) for prime in self.primes)

    def as_boolean_solution(self) -> BooleanSolution:
        """Return the plain solution object, for interoperability."""
        return BooleanSolution(tuple(prime.implicant for prime in self.primes))

    def explain(self, conditions: tuple[str, ...]) -> str:
        """Explain why each term is present."""
        lines = [f"{self.as_expression(conditions)}"]
        for prime in self.primes:
            reason = (
                "essential: the only prime covering at least one row"
                if prime.index in self.essential_indices
                else "selected among interchangeable alternatives"
            )
            lines.append(f"  {prime.as_expression(conditions)} — {reason}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MinimizationResult:
    """A complete minimisation: the chart, every minimum cover, and diagnostics."""

    chart: PrimeImplicantChart
    covers: tuple[MinimalCover, ...]
    truncated: bool

    @property
    def cost(self) -> tuple[int, int]:
        """Return the shared cost of every returned cover."""
        if not self.covers:  # pragma: no cover - covers is never empty in practice
            return (0, 0)
        return self.covers[0].cost

    @property
    def ambiguous(self) -> bool:
        """Return whether more than one cover attains the minimum."""
        return len(self.covers) > 1

    def as_boolean_solutions(self) -> tuple[BooleanSolution, ...]:
        """Return the plain solution objects, for interoperability."""
        return tuple(cover.as_boolean_solution() for cover in self.covers)

    def summary(self, conditions: tuple[str, ...]) -> str:
        """Return a human-readable account of the minimisation."""
        lines = [
            f"Configurations to cover: {len(self.chart.on_set)}",
            f"Prime implicants: {len(self.chart.primes)}",
            f"Essential primes: {len(self.chart.essential)}",
            f"Dominated primes: {len(self.chart.dominated)}",
            f"Minimum cost: {self.cost[0]} implicants, {self.cost[1]} literals",
            f"Minimum covers: {len(self.covers)}" + (" (truncated)" if self.truncated else ""),
        ]
        lines.extend(f"  {cover.as_expression(conditions)}" for cover in self.covers)
        return "\n".join(lines)


def build_chart(
    on_set: set[int],
    *,
    dont_cares: set[int] | None = None,
    width: int,
) -> PrimeImplicantChart:
    """Generate the prime implicants and assemble the chart.

    Parameters
    ----------
    on_set : set of int
        Configurations that must be covered.
    dont_cares : set of int, optional
        Logical remainders, usable but not required.
    width : int
        Number of conditions.

    Returns
    -------
    PrimeImplicantChart
        The chart, whose primes are ordered by literal count then bit pattern.
    """
    required = set(on_set)
    primes = prime_implicants(required, set() if dont_cares is None else set(dont_cares), width)
    entries = tuple(
        PrimeImplicant(
            index=index,
            implicant=implicant,
            covered=frozenset(m for m in required if implicant.covers(m)),
        )
        for index, implicant in enumerate(primes)
    )
    return PrimeImplicantChart(on_set=frozenset(required), primes=entries)


def minimize_chart(
    on_set: set[int],
    *,
    dont_cares: set[int] | None = None,
    width: int,
    max_solutions: int = 256,
) -> MinimizationResult:
    """Minimise, returning the chart and diagnostics alongside the covers.

    This is :func:`~setqca.minimize.minimize` with its reasoning exposed. The
    covers are identical; only the surrounding detail is added.

    Parameters
    ----------
    on_set : set of int
        Configurations that must be covered.
    dont_cares : set of int, optional
        Logical remainders, usable but not required.
    width : int
        Number of conditions.
    max_solutions : int, default 256
        Upper bound on the number of tied minimum covers returned.

    Returns
    -------
    MinimizationResult
        Chart, every minimum cover found, and whether the list was truncated.

    Raises
    ------
    RuntimeError
        If some configuration is covered by no prime implicant.
    """
    chart = build_chart(on_set, dont_cares=dont_cares, width=width)
    solutions = exact_minimum_covers(
        tuple(prime.implicant for prime in chart.primes),
        set(on_set),
        max_solutions=max_solutions,
    )

    by_implicant = {prime.implicant: prime for prime in chart.primes}
    essential_indices = frozenset(prime.index for prime in chart.essential)
    covers = tuple(
        MinimalCover(
            primes=tuple(by_implicant[implicant] for implicant in solution.implicants),
            essential_indices=essential_indices,
        )
        for solution in solutions
    )
    return MinimizationResult(
        chart=chart,
        covers=covers,
        truncated=len(covers) >= max_solutions,
    )
