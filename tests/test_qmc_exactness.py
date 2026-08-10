from itertools import combinations, product
from random import Random

from setqca.minimize import minimize


def _covers(pattern: tuple[int | None, ...], minterm: int, width: int) -> bool:
    bits = tuple((minterm >> shift) & 1 for shift in reversed(range(width)))
    return all(p is None or p == b for p, b in zip(pattern, bits, strict=True))


def _brute_cost(on_set: set[int], dont_cares: set[int], width: int) -> tuple[int, int]:
    off_set = set(range(2**width)) - on_set - dont_cares
    cubes: list[tuple[set[int], int]] = []
    for pattern in product((0, 1, None), repeat=width):
        covered = {m for m in range(2**width) if _covers(pattern, m, width)}
        if covered & on_set and not covered & off_set:
            cubes.append((covered & on_set, sum(p is not None for p in pattern)))

    for size in range(1, len(cubes) + 1):
        costs: list[tuple[int, int]] = []
        for choice in combinations(range(len(cubes)), size):
            covered: set[int] = set()
            literals = 0
            for index in choice:
                covered |= cubes[index][0]
                literals += cubes[index][1]
            if covered >= on_set:
                costs.append((size, literals))
        if costs:
            return min(costs)
    raise AssertionError("No brute-force cover found")


def test_qmc_matches_bruteforce_on_deterministic_small_random_tables() -> None:
    rng = Random(20260809)
    width = 3
    universe = set(range(2**width))
    for _ in range(30):
        on_set = {m for m in universe if rng.random() < 0.4}
        if not on_set:
            continue
        remaining = universe - on_set
        dont_cares = {m for m in remaining if rng.random() < 0.25}
        expected = _brute_cost(on_set, dont_cares, width)
        solution = minimize(on_set, dont_cares=dont_cares, width=width)[0]
        obtained = (len(solution.implicants), solution.literal_count)
        assert obtained == expected
