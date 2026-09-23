"""Research-style workers (full predict inside the worker), but one parent message per request."""
import multiprocessing as mp, os, sys, threading, time, numpy as np

def worker(dev, L, conn):
    from laya_apple import Laya
    from laya_apple.workload import make_request
    laya = Laya.from_pretrained("laya-typed-decisions", device=dev, local_files_only=True)
    s, q = make_request(laya.tokenizer, laya.config, L, 1, seed=0)
    for _ in range(10): laya.predict(context=s, questions=q)
    conn.send("ready")
    while True:
        m = conn.recv()
        if m is None: return
        laya.predict(context=s, questions=q); conn.send(1)

if __name__ == "__main__":
    ctx = mp.get_context("spawn"); w = {}
    for n, d, L in (("short", "ane", 128), ("long", "gpu", 1024)):
        a, b = ctx.Pipe(); p = ctx.Process(target=worker, args=(d, L, b), daemon=True); p.start(); w[n] = (a, p)
    for a, _ in w.values(): a.recv()
    def loop(n, secs, out):
        a = w[n][0]; end = time.monotonic() + secs
        while time.monotonic() < end:
            t = time.perf_counter(); a.send(1); a.recv(); out.append((time.perf_counter() - t) * 1e3)
    res = []
    for i in range(5):
        time.sleep(2); outs = {n: [] for n in w}
        ts = [threading.Thread(target=loop, args=(n, 6, outs[n])) for n in w]; [t.start() for t in ts]; [t.join() for t in ts]
        res.append(tuple(v for n in ("short", "long") for v in (round(len(outs[n]) / 6, 1), round(float(np.percentile(outs[n], 99)), 1))))
    print("ipc-per-request research workers", res, flush=True)
    for a, p in w.values(): a.send(None); p.join(10)
