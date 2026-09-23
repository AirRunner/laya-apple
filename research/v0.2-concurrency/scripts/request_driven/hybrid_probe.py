import os, threading, time, numpy as np
from laya_apple import Laya
from laya_apple.workload import make_request
M="laya-typed-decisions"
mode=os.environ["MODE"]
if mode=="ane_inline":
    ane=Laya.from_pretrained(M, device="ane", local_files_only=True)
    gpu=Laya.from_pretrained(M, device="gpu", execution="workers", local_files_only=True)
elif mode=="gpu_inline":
    ane=Laya.from_pretrained(M, device="ane", execution="workers", local_files_only=True)
    gpu=Laya.from_pretrained(M, device="gpu", local_files_only=True)
else:
    ane=Laya.from_pretrained(M, device="ane", execution="workers", local_files_only=True)
    gpu=Laya.from_pretrained(M, device="gpu", execution="workers", local_files_only=True)
reqs={"short":(ane,)+make_request(ane.tokenizer,ane.config,128,1,seed=0),"long":(gpu,)+make_request(ane.tokenizer,ane.config,1024,1,seed=0)}
def loop(name, secs, out):
    inst,s,q=reqs[name]; end=time.monotonic()+secs
    while time.monotonic()<end:
        t=time.perf_counter(); inst.predict(context=s,questions=q); out.append((time.perf_counter()-t)*1e3)
res=[]
for i in range(5):
    time.sleep(2)
    outs={n:[] for n in reqs}; ts=[threading.Thread(target=loop,args=(n,6,outs[n])) for n in reqs]
    [t.start() for t in ts]; [t.join() for t in ts]
    res.append((round(len(outs["short"])/6,1), round(float(np.percentile(outs["short"],99)),1), round(len(outs["long"])/6,1), round(float(np.percentile(outs["long"],99)),1)))
print(mode, res, flush=True)
for x in (ane,gpu): x.close()
