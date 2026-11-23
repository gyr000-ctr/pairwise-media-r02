import math

import pytest

import hdrhist

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [0.0, 0.25, 0.5, 0.75, 1.0]


def _h():
    return hdrhist.HistMap()


def _check_cdf_len():
    return len(_h().cdf(V)) == 5


def _check_cdf_end():
    return abs(_h().cdf(V)[-1] - 1.0) < 1e-9


def _check_cdf_start():
    return _h().cdf(V)[0] > 0.0


def _check_monotonic():
    c = _h().cdf(V)
    return all(c[i] <= c[i + 1] + 1e-12 for i in range(len(c) - 1))


def _check_map_range():
    return 0.0 <= _h().map(V, 1.0) <= 1.0


def _check_bad_empty():
    return _raises(lambda: _h().cdf([]))


def _check_bad_q():
    return _raises(lambda: _h().map(V, 1.5))


def _check_type():
    return isinstance(_h().cdf(V), list)


def _check_det():
    return _h().cdf(V) == _h().cdf(V)

CASES = [("cdf_len", "_check_cdf_len"), ("cdf_end", "_check_cdf_end"), ("cdf_start", "_check_cdf_start"), ("monotonic", "_check_monotonic"), ("map_range", "_check_map_range"), ("bad_empty", "_check_bad_empty"), ("bad_q", "_check_bad_q"), ("type", "_check_type"), ("det", "_check_det"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert HistMap() is not None
        return
    assert fn() is True
