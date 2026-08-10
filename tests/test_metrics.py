import pytest

from setqca import necessity, sufficiency


def test_sufficiency_requires_equal_length_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        sufficiency([1, 0, 1], [1, 0])


def test_necessity_requires_equal_length_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        necessity([1, 0, 1], [1, 0])


def test_fit_parameters_of_an_empty_set_are_zero_rather_than_undefined() -> None:
    """A cause with no membership anywhere yields 0 instead of dividing by zero."""
    fit = sufficiency([0, 0], [1, 1])
    assert fit.consistency == 0.0
    assert fit.coverage == 0.0
    assert fit.pri == 0.0


def test_sufficiency_crisp_perfect_subset() -> None:
    fit = sufficiency([1, 1, 0, 0], [1, 1, 1, 0])
    assert fit.consistency == pytest.approx(1.0)
    assert fit.coverage == pytest.approx(2 / 3)
    assert fit.pri == pytest.approx(1.0)


def test_necessity_crisp_perfect_superset() -> None:
    fit = necessity([1, 1, 1, 1], [1, 1, 0, 0])
    assert fit.consistency == pytest.approx(1.0)
    assert fit.coverage == pytest.approx(0.5)
    assert 0.0 <= fit.ron <= 1.0
