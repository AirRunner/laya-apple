Probes behind "Request-driven execution" in ../../README.md. Run from the repo root with
LAYA_APPLE_CACHE and HF_HUB_OFFLINE=1 set, e.g.

    .venv/bin/python research/v0.2-concurrency/scripts/request_driven/hetero_probe.py
    MODE=ane_inline .venv/bin/python research/v0.2-concurrency/scripts/request_driven/hybrid_probe.py

hetero_probe.py        product Laya(execution="workers"), short+long closed loops, 6 windows
hybrid_probe.py        MODE=ane_inline | gpu_inline | both_workers (separate Laya instances)
research_proc_probe.py the step-1 "processes" condition (self-driven loops in each worker)
ipc_probe.py           the same research workers, but one parent message per request
ipc_single.py          one parent process per stream (run two at once with a shared start time)

The mitigation experiments (thread QoS, spin-polling, compute-thread relay, new session,
in-worker probes) used temporary environment toggles in laya_apple/executor.py that are
not kept; their results are recorded in the README.
