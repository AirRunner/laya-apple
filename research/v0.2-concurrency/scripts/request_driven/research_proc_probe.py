import multiprocessing as mp, sys, time, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from inproc import worker
if __name__ == "__main__":
    ctx=mp.get_context("spawn"); w={}
    for name,dev,L in (("ane_short","ane",128),("gpu_long","gpu",1024)):
        a,b=ctx.Pipe(); p=ctx.Process(target=worker,args=("laya-typed-decisions",dev,L,b),daemon=True); p.start(); w[name]=(a,p)
    for a,_ in w.values(): a.recv()
    for i in range(4):
        time.sleep(2); st=time.monotonic()+0.5
        for a,_ in w.values(): a.send({"op":"run","start_at":st,"seconds":6})
        r={n:a.recv() for n,(a,_) in w.items()}
        print("research-processes", {n:(round(len(v["latency_ms"])/v["wall_s"],1), round(float(np.percentile(v["latency_ms"],99)),1)) for n,v in r.items()}, flush=True)
    for a,p in w.values(): a.send({"op":"stop"}); p.join(10)
