import math

import pytest

import curveclosest
from curveclosest import Closest

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

C = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))


def _c():
    return curveclosest.Closest()


def _check_mid():
    t = _c().t_of(*C, (0.5, 0.75))
    return abs(t - 0.5) < 1e-3


def _check_start():
    t = _c().t_of(*C, (0.0, 0.0))
    return t < 1e-3


def _check_end():
    t = _c().t_of(*C, (1.0, 0.0))
    return abs(t - 1.0) < 1e-3


def _check_range():
    t = _c().t_of(*C, (5.0, 5.0))
    return 0.0 <= t <= 1.0


def _check_distance():
    return _c().distance(C, (5.0, 5.0)) > 0.0


def _check_bad_t():
    return _raises(lambda: _c().t_of((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0)))


def _check_type():
    return isinstance(_c().t_of(*C, (0.5, 0.5)), float)


def _check_det():
    return _c().t_of(*C, (0.5, 0.5)) == _c().t_of(*C, (0.5, 0.5))

CASES = [("mid", "_check_mid"), ("start", "_check_start"), ("end", "_check_end"), ("range", "_check_range"), ("distance", "_check_distance"), ("bad_t", "_check_bad_t"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Closest() is not None
        return
    assert fn() is True
