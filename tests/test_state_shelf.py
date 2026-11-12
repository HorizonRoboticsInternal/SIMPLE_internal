"""Duplicate-row defect of the real-rig data exporter: bursty/late delivery of the 50 Hz state messages
must not produce rows with no new state. Emulates a 30 Hz camera-paced exporter loop against a
publisher whose messages are delivered in pairs every 66 ms (what a starved spin thread looks like)."""
import threading
import time

import numpy as np

from decoupled_wbc.control.utils.state_shelf import StateShelf


def _bursty_publisher(shelf, n_pairs=15, pair_gap=0.066, in_pair=0.002):
    def run():
        for i in range(n_pairs):
            shelf.push(("msg", 2 * i)); time.sleep(in_pair); shelf.push(("msg", 2 * i + 1)); time.sleep(pair_gap - in_pair)
    th = threading.Thread(target=run, daemon=True); th.start(); return th


def _old_single_slot(n_pairs=15, pair_gap=0.066, in_pair=0.002, period=1 / 30):
    """The previous exporter: one-message shelf, consume once per camera frame, no waiting."""
    slot = {"m": None}
    def run():
        for i in range(n_pairs):
            slot["m"] = 2 * i; time.sleep(in_pair); slot["m"] = 2 * i + 1; time.sleep(pair_gap - in_pair)
    th = threading.Thread(target=run, daemon=True); th.start()
    stale = 0; rows = 0; t0 = time.monotonic()
    while th.is_alive():
        m = slot["m"]; slot["m"] = None
        stale += m is None; rows += 1
        time.sleep(max(0.0, t0 + rows * period - time.monotonic()))
    return stale / max(rows, 1)


def test_bursty_delivery_no_stale_rows_with_wait():
    shelf = StateShelf(maxlen=50); th = _bursty_publisher(shelf)
    time.sleep(0.01); stale = rows = waited = 0; t0 = time.monotonic(); period = 1 / 30
    while th.is_alive():
        raw, is_new, _ = shelf.take_newest()
        if not is_new:
            if shelf.wait_for_newer(0.04):
                raw, is_new, _ = shelf.take_newest(); waited += 1
            elif not th.is_alive():
                break                                   # publisher finished: no more messages to wait for
        stale += not is_new; rows += 1
        time.sleep(max(0.0, t0 + rows * period - time.monotonic()))
    assert rows >= 12
    assert stale == 0, f"{stale}/{rows} rows had no new state despite messages existing"
    assert waited > 0, "the wait path should have been exercised by the bursty delivery"


def test_old_single_slot_reproduces_the_defect():
    frac = _old_single_slot()
    assert frac > 0.2, f"the old consume-once shelf should show many stale rows on bursty delivery, got {frac:.2f}"


def test_shelf_semantics():
    s = StateShelf(maxlen=3)
    assert s.take_newest() == (None, False, s.take_newest()[2]) or s.take_newest()[0] is None
    s.push("a"); s.push("b")
    raw, new, t = s.take_newest(); assert raw == "b" and new and t > 0
    raw, new, _ = s.take_newest(); assert raw == "b" and not new           # same message again -> not new
    s.push("c"); assert s.wait_for_newer(0.01) is True
    raw, new, _ = s.take_newest(); assert raw == "c" and new
    assert s.wait_for_newer(0.02) is False                                # nothing newer arrives
    for x in range(10): s.push(x)
    assert len(s._buf) == 3 and s.seq == 13
