"""Repeated load/close of execution="workers" must not leak
worker processes, file descriptors or parent RSS.
"""

from __future__ import annotations

import gc
import os

import psutil

from laya_apple import Laya

MODEL = "laya-typed-decisions"
CYCLES = 20


def _child_pids():
    return {c.pid for c in psutil.Process(os.getpid()).children(recursive=True)}


def test_repeated_load_close_leaks_nothing():
    proc = psutil.Process(os.getpid())
    gc.collect()
    baseline_fds = proc.num_fds()
    baseline_rss = proc.memory_info().rss
    assert not _child_pids(), "child processes present before the test even starts"

    rss_samples = []
    for _ in range(CYCLES):
        laya = Laya.from_pretrained(MODEL, device="gpu", execution="workers", local_files_only=True)
        try:
            r = laya.predict(context="The invoice was charged twice.", questions=["Is a refund requested?"])
            assert r.answers
        finally:
            laya.close()
        gc.collect()
        rss_samples.append(proc.memory_info().rss)

    assert not _child_pids(), "worker child processes leaked after close()"

    fds = proc.num_fds()
    assert fds <= baseline_fds + 16, f"file descriptor leak: baseline={baseline_fds} now={fds}"

    # Parent RSS growth across cycles should be small, not proportional to CYCLES.
    growth = rss_samples[-1] - baseline_rss
    assert growth < 300 * 1024 * 1024, f"parent RSS grew {growth / 1e6:.1f} MB over {CYCLES} cycles"
