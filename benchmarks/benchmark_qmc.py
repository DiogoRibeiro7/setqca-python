"""Reproducible benchmark for the exact Quine-McCluskey engine.

Exact minimisation is worst-case exponential in the number of conditions, so
this script is deliberately bounded. The default width range is chosen to finish
in seconds on ordinary hardware and is safe to run in CI; raise ``--max-width``
to explore where the cliff is on your machine.

Usage
-----
    python benchmarks/benchmark_qmc.py
    python benchmarks/benchmark_qmc.py --max-width 10 --density 0.3
"""

from __future__ import annotations

import argparse
from random import Random
from time import perf_counter

from setqca.minimize import minimize
from setqca.minimize.qmc import exact_minimum_covers, prime_implicants

DEFAULT_MIN_WIDTH = 4
DEFAULT_MAX_WIDTH = 8
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_MAX_WIDTH,
        help="Largest number of conditions to benchmark. Runtime grows sharply beyond 10.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.25,
        help="Probability that a minterm belongs to the on-set.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-solutions", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Run the benchmark across the requested width range."""
    args = parse_args()
    rng = Random(args.seed)

    header = (
        f"{'width':>5} {'on':>5} {'dc':>4} {'primes':>7} {'sols':>5} "
        f"{'primes_s':>10} {'chart_s':>10} {'total_s':>10}"
    )
    print(header)
    print("-" * len(header))

    for width in range(args.min_width, args.max_width + 1):
        universe = set(range(2**width))
        on_set = {m for m in universe if rng.random() < args.density}
        dont_cares = {m for m in universe - on_set if rng.random() < 0.1}

        # Time the two phases separately: prime generation is polynomial in the
        # number of minterms, while solving the chart is the exponential part.
        start = perf_counter()
        primes = prime_implicants(on_set, dont_cares, width)
        primes_elapsed = perf_counter() - start

        start = perf_counter()
        solutions = exact_minimum_covers(primes, on_set, max_solutions=args.max_solutions)
        chart_elapsed = perf_counter() - start

        print(
            f"{width:>5} {len(on_set):>5} {len(dont_cares):>4} {len(primes):>7} "
            f"{len(solutions):>5} {primes_elapsed:>10.4f} {chart_elapsed:>10.4f} "
            f"{primes_elapsed + chart_elapsed:>10.4f}"
        )


if __name__ == "__main__":
    main()
    # Keep the public entry point exercised as well as the internals above.
    assert minimize({6, 7}, width=3)[0].as_expression(("A", "B", "C")) == "A*B"
