import time
from functools import partial

from pytest import approx

from .timer import TimerManager


def test_timer():
    quant = 0.1
    bt = TimerManager()
    time.sleep(1 * quant)

    with bt.timer("total"):
        with bt.timer("load_data"):
            time.sleep(1 * quant)

        time.sleep(1 * quant)

        with bt.timer("fe"):
            time.sleep(2 * quant)

        with bt.timer("predict"):
            time.sleep(3 * quant)

    time.sleep(1 * quant)

    results = bt.get_results()
    appr = partial(approx, rel=0.01)

    assert results["total.load_data"] == appr(1 * quant)
    assert results["total.fe"] == appr(2 * quant)
    assert results["total.predict"] == appr(3 * quant)
    assert results["total"] == appr(7 * quant)
