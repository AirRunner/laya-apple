import multiprocessing as mp, sys, time, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from ipc_probe import worker
if __name__ == "__main__":
    n, d, L, start = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    ctx = mp.get_context("spawn"); a, b = ctx.Pipe(); p = ctx.Process(target=worker, args=(d, L, b), daemon=True); p.start(); a.recv()
    res = []
    for i in range(5):
        t0 = start + i * 8
        while time.time() < t0: time.sleep(0.001)
        end = time.monotonic() + 6; out = []
        while time.monotonic() < end:
            t = time.perf_counter(); a.send(1); a.recv(); out.append((time.perf_counter() - t) * 1e3)
        res.append((round(len(out) / 6, 1), round(float(np.percentile(out, 99)), 1)))
    print(n, res, flush=True); a.send(None); p.join(10)
