"""공통 도구: 매니페스트·입력 읽기, CUDA 런타임 (ctypes, torch/pycuda 불필요), TensorRT 실행기, 전력 표본 추출기.
numpy + tensorrt (JetPack 기본 제공) 만 필요하다. onnxruntime 실행기 (--backend ort) 는 PC 에서 스크립트 점검용이다."""
import ctypes, glob, json, os, threading, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(ROOT, "engines")


def load_manifest():
    return json.load(open(os.path.join(ROOT, "manifest.json")))


def load_inputs(T, n_pad=None):
    """frames_T{T}.npz -> (float32 [N(+pad), T, 2, 32, 32], int labels [1000]). n_pad 이면 앞에서부터 되풀이해 그 길이로 채운다."""
    d = np.load(os.path.join(ROOT, "data", f"frames_T{T}.npz"))
    x = d["x"].astype(np.float32); y = d["y"].astype(np.int64)
    if n_pad and n_pad > len(x):
        x = np.concatenate([x, x[: n_pad - len(x)]])
    return np.ascontiguousarray(x), y


def load_expected(name):
    d = np.load(os.path.join(ROOT, "data", f"expected_{name}.npz"))
    return d["logits"], d["pred"].astype(np.int64), d["layer_spikes"].astype(np.int64)


# ---------------------------------------------------------------- CUDA 런타임 (ctypes)
class Cuda:
    H2D, D2H = 1, 2

    def __init__(self):
        last = None
        for nm in ("libcudart.so.12", "libcudart.so", "/usr/local/cuda/lib64/libcudart.so", "/usr/local/cuda-12.6/lib64/libcudart.so"):
            try:
                self.lib = ctypes.CDLL(nm); break
            except OSError as e:
                last = e
        else:
            raise RuntimeError(f"libcudart 를 찾지 못했다: {last}")
        L = self.lib; vp = ctypes.c_void_p
        L.cudaMalloc.argtypes = [ctypes.POINTER(vp), ctypes.c_size_t]
        L.cudaFree.argtypes = [vp]
        L.cudaHostAlloc.argtypes = [ctypes.POINTER(vp), ctypes.c_size_t, ctypes.c_uint]
        L.cudaFreeHost.argtypes = [vp]
        L.cudaMemcpyAsync.argtypes = [vp, vp, ctypes.c_size_t, ctypes.c_int, vp]
        L.cudaStreamCreate.argtypes = [ctypes.POINTER(vp)]
        L.cudaStreamSynchronize.argtypes = [vp]
        L.cudaStreamDestroy.argtypes = [vp]
        L.cudaEventCreate.argtypes = [ctypes.POINTER(vp)]
        L.cudaEventRecord.argtypes = [vp, vp]
        L.cudaEventSynchronize.argtypes = [vp]
        L.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), vp, vp]
        L.cudaEventDestroy.argtypes = [vp]

    def chk(self, rc, what=""):
        if rc != 0:
            raise RuntimeError(f"CUDA 오류 {rc} {what}")

    def malloc(self, n):
        p = ctypes.c_void_p(); self.chk(self.lib.cudaMalloc(ctypes.byref(p), n), "cudaMalloc"); return p.value

    def host_alloc(self, n):
        p = ctypes.c_void_p(); self.chk(self.lib.cudaHostAlloc(ctypes.byref(p), n, 0), "cudaHostAlloc"); return p.value

    def stream(self):
        p = ctypes.c_void_p(); self.chk(self.lib.cudaStreamCreate(ctypes.byref(p))); return p.value

    def event(self):
        p = ctypes.c_void_p(); self.chk(self.lib.cudaEventCreate(ctypes.byref(p))); return p.value

    def memcpy(self, dst, src, n, kind, stream):
        self.chk(self.lib.cudaMemcpyAsync(dst, src, n, kind, stream), "memcpy")

    def sync(self, stream):
        self.chk(self.lib.cudaStreamSynchronize(stream), "sync")

    def record(self, ev, stream):
        self.chk(self.lib.cudaEventRecord(ev, stream))

    def elapsed_ms(self, a, b):
        self.chk(self.lib.cudaEventSynchronize(b))
        t = ctypes.c_float(); self.chk(self.lib.cudaEventElapsedTime(ctypes.byref(t), a, b)); return t.value

    def free(self, p):
        self.lib.cudaFree(p)

    def host_free(self, p):
        self.lib.cudaFreeHost(p)

    def destroy(self, stream=None, events=()):
        for e in events: self.lib.cudaEventDestroy(e)
        if stream: self.lib.cudaStreamDestroy(stream)

    def np_view(self, ptr, dtype, shape):
        n = int(np.prod(shape)); ct = {np.float32: ctypes.c_float}[dtype]
        return np.ctypeslib.as_array((ct * n).from_address(ptr)).reshape(shape)


# ---------------------------------------------------------------- 실행기
class TRTRunner:
    """엔진 하나. 입력 전체를 pinned 메모리에 올려 두고 (데이터 읽기·형변환은 측정 밖) 표본/배치 단위로 H2D -> 실행 -> D2H -> 동기화."""

    def __init__(self, engine_path, B, x_all):
        import tensorrt as trt
        self.trt, self.B = trt, B
        self.cu = cu = Cuda()
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = rt.deserialize_cuda_engine(open(engine_path, "rb").read())
        if self.engine is None:
            raise RuntimeError(f"엔진을 읽지 못했다: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.stream = cu.stream(); self.ev0, self.ev1 = cu.event(), cu.event()
        self.n = len(x_all); self.per = int(np.prod(x_all.shape[1:])) * 4
        self.h_in_ptr = cu.host_alloc(x_all.nbytes)
        self.h_in = cu.np_view(self.h_in_ptr, np.float32, x_all.shape); self.h_in[...] = x_all
        self.io = {}
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i); shp = tuple(self.engine.get_tensor_shape(nm))
            assert self.engine.get_tensor_dtype(nm) == trt.float32, (nm, self.engine.get_tensor_dtype(nm))
            is_in = self.engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT
            nbytes = int(np.prod(shp)) * 4
            d = cu.malloc(nbytes); self.ctx.set_tensor_address(nm, d)
            ent = dict(shape=shp, nbytes=nbytes, dev=d, input=is_in)
            if not is_in:
                ent["host_ptr"] = cu.host_alloc(self.n // B * nbytes)
                ent["host"] = cu.np_view(ent["host_ptr"], np.float32, (self.n // B,) + shp)
            self.io[nm] = ent
        self.in_name = [k for k, v in self.io.items() if v["input"]][0]
        assert self.io[self.in_name]["shape"][0] == B and self.n % B == 0, (self.io[self.in_name]["shape"], B, self.n)
        self.outs = [k for k, v in self.io.items() if not v["input"]]
        self.has_spk = "layer_spikes" in self.outs

    def run(self, bi):
        """배치 bi (표본 bi*B ... ) 처리, 출력은 host 버퍼 bi 번째 칸에 남는다. 호스트 시간 (H2D+실행+D2H+동기화) 을 위해 호출 측이 perf_counter 로 잰다."""
        cu, st = self.cu, self.stream
        i = self.io[self.in_name]
        cu.memcpy(i["dev"], self.h_in_ptr + bi * self.B * self.per, i["nbytes"], cu.H2D, st)
        self.ctx.execute_async_v3(st)
        for nm in self.outs:
            o = self.io[nm]; cu.memcpy(o["host_ptr"] + bi * o["nbytes"], o["dev"], o["nbytes"], cu.D2H, st)
        cu.sync(st)

    def run_gpu_ms(self, bi):
        """실행 커널만의 GPU 시간 (cudaEvent). H2D·D2H 제외."""
        cu, st = self.cu, self.stream
        i = self.io[self.in_name]
        cu.memcpy(i["dev"], self.h_in_ptr + bi * self.B * self.per, i["nbytes"], cu.H2D, st)
        cu.record(self.ev0, st); self.ctx.execute_async_v3(st); cu.record(self.ev1, st)
        cu.sync(st)
        return cu.elapsed_ms(self.ev0, self.ev1)

    def output(self, name):
        h = self.io[name]["host"]; return h.reshape((-1,) + h.shape[2:])

    def close(self):
        cu = self.cu
        for v in self.io.values():
            cu.free(v["dev"])
            if "host_ptr" in v: cu.host_free(v["host_ptr"])
        cu.host_free(self.h_in_ptr); cu.destroy(self.stream, (self.ev0, self.ev1))
        del self.ctx, self.engine


class ORTRunner:
    """PC 점검용 (onnxruntime CPU). 같은 인터페이스."""

    def __init__(self, onnx_path, B, x_all):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.log_severity_level = 3
        self.s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"]); self.B = B; self.x = x_all
        self.outs = [o.name for o in self.s.get_outputs()]; self.has_spk = "layer_spikes" in self.outs
        self.n = len(x_all); self.h = {nm: np.zeros((self.n // B,) + tuple(o.shape), np.float32) for nm, o in zip(self.outs, self.s.get_outputs())}

    def run(self, bi):
        r = self.s.run(None, {"input": self.x[bi * self.B:(bi + 1) * self.B]})
        for nm, v in zip(self.outs, r): self.h[nm][bi] = v

    def run_gpu_ms(self, bi):
        return float("nan")

    def output(self, name):
        h = self.h[name]; return h.reshape((-1,) + h.shape[2:])

    def close(self):
        pass


# ---------------------------------------------------------------- 전력 (INA3221)
RAILS = ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC")


def find_power_rails():
    """{레일: (전압 파일, 전류 파일)}. 환경변수 EIP_HWMON_DIR 로 폴더를 지정하면 (PC 점검용) 거기를 쓴다."""
    dirs = [os.environ["EIP_HWMON_DIR"]] if os.environ.get("EIP_HWMON_DIR") else sorted(glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"))
    for d in dirs:
        rails = {}
        for k in range(1, 4):
            lf = os.path.join(d, f"in{k}_label")
            if not os.path.exists(lf):
                continue
            lab = open(lf).read().strip()
            vf, cf = os.path.join(d, f"in{k}_input"), os.path.join(d, f"curr{k}_input")
            if lab in RAILS and os.path.exists(vf) and os.path.exists(cf):
                rails[lab] = (vf, cf)
        if len(rails) == len(RAILS):
            return rails
    return None


class PowerSampler(threading.Thread):
    """레일마다 전압 (mV) x 전류 (mA) / 1000 = 전력 (mW) 을 interval 간격으로 기록한다."""

    def __init__(self, rails, interval=0.010):
        super().__init__(daemon=True)
        self.rails, self.interval, self.stop_ev = rails, interval, threading.Event()
        self.t, self.p = [], {k: [] for k in rails}

    def run(self):
        fs = {k: (open(v, "r"), open(c, "r")) for k, (v, c) in self.rails.items()}
        nxt = time.perf_counter()
        while not self.stop_ev.is_set():
            t = time.perf_counter(); row = {}
            for k, (fv, fc) in fs.items():
                fv.seek(0); fc.seek(0)
                row[k] = int(fv.read()) * int(fc.read()) / 1000.0
            self.t.append(t)
            for k in row: self.p[k].append(row[k])
            nxt += self.interval
            dt = nxt - time.perf_counter()
            if dt > 0: time.sleep(dt)
            else: nxt = time.perf_counter()
        for a, b in fs.values(): a.close(); b.close()

    def stop(self):
        self.stop_ev.set(); self.join()

    def window(self, t0, t1):
        t = np.array(self.t); m = (t >= t0) & (t <= t1)
        return {k: np.array(v)[m] for k, v in self.p.items()}


def stats(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), std=float(a.std()), n=int(a.size)) if a.size else dict(mean=float("nan"), std=float("nan"), n=0)
