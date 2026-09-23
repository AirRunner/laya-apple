"""Repeated inline predict calls on one Laya instance must not
leak memory in the parent process."""

from __future__ import annotations

import gc
import os

import psutil

from laya_apple import Laya

MODEL = "laya-typed-decisions"
CALLS = 2000
SAMPLE_EVERY = 200


def test_repeated_inline_predict_memory_growth_is_bounded():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    laya = Laya.from_pretrained(MODEL, device="gpu", execution="inline", local_files_only=True)
    proc = psutil.Process(os.getpid())
    try:
        context = "The customer was charged twice for the same invoice and is frustrated."
        questions = {"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}}

        gc.collect()
        baseline_rss = proc.memory_info().rss
        samples = []
        for i in range(CALLS):
            laya.predict(context=context, questions=questions)
            if i % SAMPLE_EVERY == 0:
                gc.collect()
                samples.append(proc.memory_info().rss)

        gc.collect()
        final_rss = proc.memory_info().rss
        growth = final_rss - baseline_rss
        assert growth < 300 * 1024 * 1024, (
            f"parent RSS grew {growth / 1e6:.1f} MB over {CALLS} inline predict calls; samples={samples}"
        )
    finally:
        laya.close()
