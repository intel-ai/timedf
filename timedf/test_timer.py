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

    assert_timer_results(results, quant, appr)


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

        assert_timer_results(results, quant, appr)
        tm.reset()


def assert_timer_results(results, quant, appr):
    assert results.pop("total.load_data") == appr(1 * quant)
    assert results.pop("total.fe") == appr(2 * quant)
    assert results.pop("total.predict") == appr(3 * quant)
    assert results.pop("total") == appr(7 * quant)
    assert len(results) == 0
