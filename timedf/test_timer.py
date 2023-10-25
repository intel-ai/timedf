import time
from functools import partial

from pytest import approx, raises

from .timer import TimerManager


def test_timer():
    quant = 0.1
    bt = TimerManager()
    time.sleep(1 * quant)

    with bt.timeit("total"):
        with bt.timeit("load_data"):
            time.sleep(1 * quant)

        time.sleep(1 * quant)

        with bt.timeit("fe"):
            time.sleep(2 * quant)

        with bt.timeit("predict"):
            time.sleep(3 * quant)

    time.sleep(1 * quant)

    results = bt.get_results()
    appr = partial(approx, rel=0.01)

    assert results["total.load_data"] == appr(1 * quant)
    assert results["total.fe"] == appr(2 * quant)
    assert results["total.predict"] == appr(3 * quant)
    assert results["total"] == appr(7 * quant)
    assert len(results) == 4


def test_timer_acc():
    quant = 0.1
    bt = TimerManager()
    time.sleep(1 * quant)

    with bt.timeit("total"):
        with bt.timeit("load_data"):
            time.sleep(1 * quant)

        time.sleep(1 * quant)

        with bt.timeit("fe"):
            time.sleep(2 * quant)

        for i in range(10):
            with bt.timeit("predict"):
                time.sleep(3 * quant)
                with bt.timeit("minor"):
                    time.sleep(quant)

    time.sleep(1 * quant)

    results = bt.get_results()
    appr = partial(approx, rel=0.01)

    assert results["total.load_data"] == appr(1 * quant)
    assert results["total.fe"] == appr(2 * quant)
    assert results["total.predict"] == appr(40 * quant)
    assert results["total.predict.minor"] == appr(10 * quant)
    assert results["total"] == appr(44 * quant)
    assert len(results) == 5


def test_timer_state_noname():
    bt = TimerManager()
    with raises(ValueError):
        with bt:
            pass


def test_timer_state_reopen():
    bt = TimerManager()
    bt.timeit("b")
    with raises(ValueError):
        bt.timeit("b")


def test_timer_reset():
    quant = 0.1
    tm = TimerManager()
    time.sleep(1 * quant)

    for i in range(3):
        with tm.timeit("total"):
            with tm.timeit("load_data"):
                time.sleep(1 * quant)

            time.sleep(1 * quant)

            with tm.timeit("fe"):
                time.sleep(2 * quant)

            with tm.timeit("predict"):
                time.sleep(3 * quant)

        time.sleep(1 * quant)

        results = tm.get_results()
        appr = partial(approx, rel=0.01)

        assert results["total.load_data"] == appr(1 * quant)
        assert results["total.fe"] == appr(2 * quant)
        assert results["total.predict"] == appr(3 * quant)
        assert results["total"] == appr(7 * quant)
        assert len(results) == 4
        tm.reset()
