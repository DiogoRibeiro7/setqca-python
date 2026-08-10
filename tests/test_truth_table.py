import pandas as pd
import pytest

from setqca import build_truth_table


def test_crisp_truth_table_codes_positive_negative_and_remainder() -> None:
    data = pd.DataFrame(
        {
            "case": ["a", "b", "c"],
            "A": [1, 1, 0],
            "B": [1, 0, 0],
            "Y": [1, 1, 0],
        }
    )
    tt = build_truth_table(
        data,
        outcome="Y",
        conditions=["A", "B"],
        inclusion_cutoff=1.0,
        frequency_cutoff=1,
        case_id="case",
    )
    assert tt.positive_minterms == {2, 3}
    assert 1 in tt.remainder_minterms
    assert 0 in tt.negative_minterms


def test_crossover_case_rejected_by_default() -> None:
    data = pd.DataFrame({"A": [0.5], "Y": [0.8]})
    with pytest.raises(ValueError, match="crossover"):
        build_truth_table(data, outcome="Y", conditions=["A"])
