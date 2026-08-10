from setqca.minimize import minimize, prime_implicants


def test_qmc_reduces_adjacent_minterms() -> None:
    # AB~C + ABC = AB
    solutions = minimize({6, 7}, width=3)
    assert len(solutions) == 1
    assert solutions[0].as_expression(("A", "B", "C")) == "A*B"


def test_qmc_uses_dont_cares_for_parsimony() -> None:
    # Positive rows 110/111 plus remainder 100/101 permit A.
    solutions = minimize({6, 7}, dont_cares={4, 5}, width=3)
    assert solutions[0].as_expression(("A", "B", "C")) == "A"


def test_prime_implicants_do_not_include_pure_remainders() -> None:
    primes = prime_implicants({7}, {0}, width=3)
    assert all(any(item.covers(m) for m in {7}) for item in primes)
