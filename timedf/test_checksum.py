import time
from functools import partial

from pytest import approx, raises

import numpy as np

from .checksum import ChecksumStore


def test_checksum():
    quant = 0.1
    cs = ChecksumStore()
    time.sleep(1 * quant)

    array = np.arange(100) / 10

    with cs.check("sum") as c:
        time.sleep(quant)

        c.set(array.sum())

    with cs.check("mean") as c:
        time.sleep(2 * quant)
        c.set(array.mean())

    appr = partial(approx, rel=0.01)
    res = cs.get_results()
    assert res["sum"]["duration_s"] == appr(quant)
    assert res["mean"]["duration_s"] == appr(2 * quant)

    assert res["sum"]["value"] == 495
    assert res["mean"]["value"] == appr(4.95)
