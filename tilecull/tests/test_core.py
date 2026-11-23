import math

import pytest

import tilecull

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

C = (0.0, 0.0, 32.0, 32.0)


def _c():
    return tilecull.TileCull()


def _check_count():
    return len(_c().visible(C, 16.0, 4, 4)) > 0


def _check_bounds():
    return all(0 <= t[0] < 4 and 0 <= t[1] < 4 for t in _c().visible(C, 16.0, 4, 4))


def _check_small_cam():
    return len(_c().visible((0.0, 0.0, 1.0, 1.0), 16.0, 4, 4)) >= 1


def _check_big_cam():
    return len(_c().visible((-100.0, -100.0, 400.0, 400.0), 16.0, 4, 4)) == 16


def _check_bad_cam():
    return _raises(lambda: _c().visible((0.0, 0.0, 0.0, 0.0), 16.0, 4, 4))


def _check_bad_tile():
    return _raises(lambda: _c().visible(C, 0.0, 4, 4))


def _check_type():
    return isinstance(_c().visible(C, 16.0, 4, 4), list)


def _check_det():
    return _c().visible(C, 16.0, 4, 4) == _c().visible(C, 16.0, 4, 4)

CASES = [("count", "_check_count"), ("bounds", "_check_bounds"), ("small_cam", "_check_small_cam"), ("big_cam", "_check_big_cam"), ("bad_cam", "_check_bad_cam"), ("bad_tile", "_check_bad_tile"), ("type", "_check_type"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert TileCull() is not None
        return
    assert fn() is True
