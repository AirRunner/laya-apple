import os, sys, threading, time, numpy as np
from laya_apple import Laya
from laya_apple.workload import make_request
if os.environ.get("SWITCH"): sys.setswitchinterval(float(os.environ["SWITCH"]))
M=os.environ.get("M","laya-typed-decisions")
l=Laya.from_pretrained(M, execution="workers", local_files_only=True)
reqs={"short":make_request(l.tokenizer,l.config,128,1,seed=0),"long":make_request(l.tokenizer,l.config,1024,1,seed=0)}
def loop(name, secs, out):
    if os.environ.get("CLIENTQOS"):
        import ctypes; ctypes.CDLL("/usr/lib/libSystem.dylib").pthread_set_qos_class_self_np(ctypes.c_uint(0x21), ctypes.c_int(0))
    s,q=reqs[name]; end=time.monotonic()+secs
    while time.monotonic()<end:
        t=time.perf_counter(); r=l.predict(context=s,questions=q); out.append(((time.perf_counter()-t)*1e3, r.runtime.device_ms))
res=[]
for i in range(6):
    time.sleep(2)
    outs={n:[] for n in reqs}; ts=[threading.Thread(target=loop,args=(n,6,outs[n])) for n in reqs]
    [t.start() for t in ts]; [t.join() for t in ts]
    res.append((round(len(outs["short"])/6,1), round(float(np.percentile([x[0] for x in outs["short"]],99)),1), round(len(outs["long"])/6,1), round(float(np.percentile([x[0] for x in outs["long"]],99)),1)))
print(os.environ.get("TAG"), res, flush=True)
l.close()
