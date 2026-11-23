"""TileCull 的自测：与暴力逐瓦片判定比对、边界、性能、并发、错误输入。"""

import math
import random
import threading
import time

import pytest

import tilecull


def brute(cam, ts, w, h):
    x0, y0, x1, y1 = cam
    out = []
    for j in range(h):
        for i in range(w):
            tx0 = i * ts
            ty0 = j * ts
            if tx0 < x1 and tx0 + ts > x0 and ty0 < y1 and ty0 + ts > y0:
                out.append((i, j))
    return out


def test_matches_brute_force_random():
    rng = random.Random(1234)
    c = tilecull.TileCull()
    for _ in range(300):
        w = rng.randint(1, 40)
        h = rng.randint(1, 40)
        ts = rng.choice([0.5, 1.0, 2.5, 16.0, 3.0])
        # 让相机边界经常落在瓦片边界上，专门触发归属判定。
        def coord(lim, extra):
            return rng.choice(
                [rng.uniform(-lim, lim), rng.randrange(-2, int(lim) + 3) * ts,
                 rng.uniform(0.0, lim)]
            ) + extra
        x0 = coord(w * ts, 0.0)
        x1 = x0 + rng.choice([ts, 2 * ts, rng.uniform(0.0, w * ts), 1e-9])
        y0 = rng.uniform(-h * ts, 2 * h * ts)
        y1 = y0 + rng.choice([ts, rng.uniform(0.0, h * ts), 1e-9])
        if not (x0 < x1 and y0 < y1):
            continue
        assert c.visible((x0, y0, x1, y1), ts, w, h) == brute((x0, y0, x1, y1), ts, w, h)


def test_boundary_tiles_excluded_when_touching():
    c = tilecull.TileCull()
    # 相机恰好覆盖 (1,1) 瓦片，仅与相邻瓦片相切 -> 只有它自己。
    assert c.visible((16.0, 16.0, 32.0, 32.0), 16.0, 4, 4) == [(1, 1)]
    assert c.visible((0.0, 0.0, 16.0, 16.0), 16.0, 4, 4) == [(0, 0)]


def test_camera_inside_tile_not_missed():
    c = tilecull.TileCull()
    assert c.visible((17.0, 18.0, 17.5, 18.5), 16.0, 4, 4) == [(1, 1)]
    # 跨在瓦片接缝上的极小相机 -> 两块都可见。
    assert c.visible((15.9, 15.9, 16.1, 16.1), 16.0, 4, 4) == [
        (0, 0), (1, 0), (0, 1), (1, 1)
    ]


def test_camera_outside_grid():
    c = tilecull.TileCull()
    assert c.visible((-10.0, -10.0, 0.0, 0.0), 16.0, 4, 4) == []
    assert c.visible((64.0, 0.0, 100.0, 32.0), 16.0, 4, 4) == []
    assert c.visible((-1.0, -1.0, 65.0, 65.0), 16.0, 4, 4) == brute(
        (-1.0, -1.0, 65.0, 65.0), 16.0, 4, 4
    )


def test_deterministic_and_rows_major():
    c = tilecull.TileCull()
    a = c.visible((0.0, 0.0, 32.0, 32.0), 16.0, 4, 4)
    assert a == c.visible((0.0, 0.0, 32.0, 32.0), 16.0, 4, 4)
    assert a == [(i, j) for j in range(2) for i in range(2)]


@pytest.mark.parametrize("cam", [
    (0.0, 0.0, 0.0, 10.0),
    (0.0, 0.0, 10.0, 0.0),
    (10.0, 0.0, 0.0, 10.0),
    (0.0, 10.0, 10.0, 0.0),
    (float("nan"), 0.0, 1.0, 1.0),
    (0.0, float("inf"), 1.0, 1.0),
    (0.0, 0.0, float("-inf"), 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0),
    (0.0, 0.0, 1.0, 1.0, 1.0),
])
def test_bad_cam_raises_value_error(cam):
    with pytest.raises(ValueError):
        tilecull.TileCull().visible(cam, 16.0, 4, 4)


@pytest.mark.parametrize("cam", [None, 3.0, "abcd", [1, 2, 3, "x"], (True, 0, 1, 1)])
def test_bad_cam_raises_type_error(cam):
    with pytest.raises(TypeError):
        tilecull.TileCull().visible(cam, 16.0, 4, 4)


@pytest.mark.parametrize("ts", [0.0, -1.0, float("inf"), float("nan")])
def test_bad_tile_size_raises(ts):
    with pytest.raises(ValueError):
        tilecull.TileCull().visible((0.0, 0.0, 1.0, 1.0), ts, 4, 4)


@pytest.mark.parametrize("w,h", [(0, 4), (4, 0), (-1, 4), (4, -1), (2.5, 4)])
def test_bad_dims_raise(w, h):
    with pytest.raises(ValueError):
        tilecull.TileCull().visible((0.0, 0.0, 1.0, 1.0), 16.0, w, h)


def test_bad_tile_size_type():
    with pytest.raises(TypeError):
        tilecull.TileCull().visible((0.0, 0.0, 1.0, 1.0), "16", 4, 4)


def test_performance_100k_tiles():
    c = tilecull.TileCull()
    w, h = 400, 250  # 100_000 瓦片
    ts = 1.0
    cam = (-0.5, -0.5, w * ts + 0.5, h * ts + 0.5)
    t0 = time.perf_counter()
    out = c.visible(cam, ts, w, h)
    elapsed = time.perf_counter() - t0
    assert len(out) == 100_000
    assert elapsed < 0.3, f"100k tiles took {elapsed * 1000:.1f} ms"

    # 小视口也应远低于上界。
    t0 = time.perf_counter()
    c.visible((100.0, 100.0, 140.0, 130.0), ts, w, h)
    assert time.perf_counter() - t0 < 0.3


def test_concurrent_calls_are_consistent():
    c = tilecull.TileCull()
    cases = [
        ((0.0, 0.0, 32.0, 32.0), 16.0, 4, 4),
        ((-100.0, -100.0, 400.0, 400.0), 16.0, 4, 4),
        ((17.0, 18.0, 17.5, 18.5), 16.0, 4, 4),
        ((0.0, 0.0, 1e6, 1e6), 16.0, 4, 4),
    ]
    expected = [c.visible(*args) for args in cases]
    errors = []

    def worker():
        try:
            for _ in range(200):
                for args, exp in zip(cases, expected):
                    got = c.visible(*args)
                    if got != exp:
                        errors.append((args, got, exp))
                        return
                    got.append((0, 0))  # 修改返回值不得影响后续调用
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


@pytest.mark.parametrize("args", [
    # 巨大瓦片尺寸：网格世界范围会溢出为 inf。
    ((0.0, 0.0, 5.0, 5.0), 1e308, 2, 2),
    ((1.5e308, 0.0, 1.7e308, 5.0), 1e308, 2, 2),
    # 极小瓦片尺寸 + 极大相机坐标。
    ((-1e300, -1e300, 1e300, 1e300), 1e-300, 1, 1),
    # 相机远离网格但跨越到网格内。
    ((-1e10, -1e10, 3.0, 3.0), 1.0, 8, 8),
    ((7.0, 7.0, 1e10, 1e10), 1.0, 8, 8),
    ((2.0, 2.0, 2.0 + 1e-12, 2.0 + 1e-12), 0.25, 16, 16),
])
def test_extreme_values_match_brute(args):
    got = tilecull.TileCull().visible(*args)
    assert got == brute(*args)


def test_accepts_integers_and_int_tile_size():
    c = tilecull.TileCull()
    assert c.visible((0, 0, 32, 32), 16, 4, 4) == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert c.visible((0.0, 0.0, 32.0, 32.0), 16, 4, 4) == c.visible(
        (0, 0, 32, 32), 16.0, 4, 4
    )


def test_returned_list_is_independent_mutation():
    c = tilecull.TileCull()
    a = c.visible((-100.0, -100.0, 400.0, 400.0), 16.0, 4, 4)
    assert len(a) == 16
    a.clear()
    assert len(c.visible((-100.0, -100.0, 400.0, 400.0), 16.0, 4, 4)) == 16
