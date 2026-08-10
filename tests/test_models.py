import pandas as pd

from setqca import CSQCA, FSQCA


def test_csqca_end_to_end() -> None:
    # Y is present whenever A is present. One unobserved A=0,B=1 row remains.
    data = pd.DataFrame(
        {
            "case": ["c1", "c2", "c3"],
            "A": [1, 1, 0],
            "B": [1, 0, 0],
            "Y": [1, 1, 0],
        }
    )
    result = CSQCA().fit(data, outcome="Y", conditions=["A", "B"], case_id="case")
    assert result.conservative[0].expression(result.conditions) == "A"
    assert result.parsimonious[0].expression(result.conditions) == "A"
    assert result.conservative[0].fit.consistency == 1.0


def test_fsqca_end_to_end() -> None:
    data = pd.DataFrame(
        {
            "A": [0.9, 0.8, 0.2, 0.1],
            "B": [0.9, 0.2, 0.8, 0.1],
            "Y": [0.95, 0.85, 0.2, 0.1],
        }
    )
    result = FSQCA(consistency=0.8, frequency=1).fit(data, outcome="Y", conditions=["A", "B"])
    assert result.truth_table.positive_minterms
    assert result.conservative
