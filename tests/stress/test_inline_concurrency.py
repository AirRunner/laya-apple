"""Inline mode holds a per-instance lock, so many threads
calling predict concurrently on one inline Laya must be as safe as calling it
sequentially — every answer equal to the sequential reference.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from laya_apple import Laya

MODEL = "laya-typed-decisions"
THREADS = 16
CALLS = 64


def test_concurrent_inline_predict_matches_sequential_reference():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    laya = Laya.from_pretrained(MODEL, device="gpu", execution="inline", local_files_only=True)
    try:
        context = "The customer was charged twice for the same invoice and is frustrated."
        questions = {"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}}
        reference = laya.predict(context=context, questions=questions).answers

        def call(_):
            return laya.predict(context=context, questions=questions).answers

        with ThreadPoolExecutor(THREADS) as pool:
            results = list(pool.map(call, range(CALLS)))

        assert all(r == reference for r in results)
    finally:
        laya.close()
