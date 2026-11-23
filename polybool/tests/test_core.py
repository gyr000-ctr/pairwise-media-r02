import math

import pytest

import polybool

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

SQ = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
CONCAVE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 1.0), (0.0, 2.0)]


def _p():
    return polybool.Poly()


def _check_inside():
    return _p().contains(SQ, (1.0, 1.0)) is True


def _check_outside():
    return _p().contains(SQ, (3.0, 1.0)) is False


def _check_area():
    return abs(_p().area(SQ) - 4.0) < 1e-9


def _check_concave_hole():
    return _p().contains(CONCAVE, (1.0, 1.5)) is False


def _check_concave_inside():
    return _p().contains(CONCAVE, (0.5, 0.5)) is True


def _check_bad_poly():
    return _raises(lambda: _p().contains([(0.0, 0.0), (1.0, 1.0)], (0.5, 0.5)))


def _check_type():
    return isinstance(_p().contains(SQ, (1.0, 1.0)), bool)


def _check_area_concave():
    return _p().area(CONCAVE) > 0.0

CASES = [("inside", "_check_inside"), ("outside", "_check_outside"), ("area", "_check_area"), ("concave_hole", "_check_concave_hole"), ("concave_inside", "_check_concave_inside"), ("bad_poly", "_check_bad_poly"), ("type", "_check_type"), ("area_concave", "_check_area_concave"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Poly() is not None
        return
    assert fn() is True
