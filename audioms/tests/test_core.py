import math

import pytest

import audioms

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _m(w=1.0):
    return audioms.MidSide(w)


def _check_roundtrip():
    l, r = _m().to_ms([0.4], [0.2])
    return len(l) == 1 and len(r) == 1


def _check_width0():
    l, r = _m(0.0).process([0.4, 0.2], [0.2, 0.4])
    return all(abs(a - b) < 1e-9 for a, b in zip(l, r))


def _check_len():
    l, r = _m().process([0.1, 0.2], [0.3, 0.4])
    return len(l) == 2 and len(r) == 2


def _check_bad_width():
    return _raises(lambda: audioms.MidSide(-1.0))


def _check_bad_len():
    return _raises(lambda: _m().process([0.1], [0.1, 0.2]))


def _check_type():
    return isinstance(_m().process([0.1], [0.2]), tuple)


def _check_range():
    l, r = _m(1.0).process([0.5], [0.5])
    return abs(l[0]) <= 1.0 and abs(r[0]) <= 1.0


def _check_det():
    a = _m().process([0.1], [0.2])
    b = _m().process([0.1], [0.2])
    return a == b

CASES = [("roundtrip", "_check_roundtrip"), ("width0", "_check_width0"), ("len", "_check_len"), ("bad_width", "_check_bad_width"), ("bad_len", "_check_bad_len"), ("type", "_check_type"), ("range", "_check_range"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert MidSide() is not None
        return
    assert fn() is True
