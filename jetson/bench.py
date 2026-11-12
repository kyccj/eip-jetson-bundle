#!/usr/bin/env python3
"""Jetson 측정: 정확도 · 지연 · 처리량 · 층별 스파이크 대조 · 전력/에너지. numpy + tensorrt 만 필요, sudo 불필요.

작업 (--tasks):  acc latency throughput spikes power     (기본: 전부)
  acc         배치 1 엔진으로 1000장 정확도, 원래 모델의 예측·로짓과 대조
  latency     배치 1, 준비 50회 버리고 1000장, 표본당 ms (호스트 시간 = H2D + 실행 + D2H + 동기화; 데이터는 미리 메모리에 올려 둠) + 커널만의 GPU ms
  throughput  배치 16, 1000장 (1008 로 채움) 반복, ms/표본
  spikes      b1_spk 엔진으로 층별 스파이크 합을 원래 모델과 대조
  power       유휴 --idle_s 초 -> 1000장 연속 처리 (--min_run_s 초 이상 되풀이) 를 --power_repeats 회. 배치 1 과 배치 16 둘 다.
              동적 전력 = 처리 구간 평균 - 유휴 평균 (양쪽 맨 앞 1초는 버림). 표본당 에너지 = 동적 전력 x 표본당 시간.
"""
import argparse, json, os, platform, subprocess, sys, time, socket
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eipcommon import ROOT, ENGINE_DIR, load_manifest, load_inputs, load_expected, TRTRunner, ORTRunner, find_power_rails, PowerSampler, stats, RAILS

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=None, help="기본: manifest 의 전부 (--only 와 같음)")
ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16"], choices=["fp32", "fp16"])
ap.add_argument("--tasks", nargs="*", default=["acc", "latency", "throughput", "spikes", "power"], choices=["acc", "latency", "throughput", "spikes", "power"])
ap.add_argument("--backend", default="trt", choices=["trt", "ort"], help="ort = PC 에서 스크립트 점검용 (onnxruntime CPU)")
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--idle_s", type=float, default=60.0)
ap.add_argument("--min_run_s", type=float, default=30.0)
ap.add_argument("--power_repeats", type=int, default=3)
ap.add_argument("--power_batches", nargs="*", type=int, default=[1, 16], choices=[1, 16])
ap.add_argument("--out", default=None, help="결과 json (기본 results/bench_<시각>.json)")
ap.add_argument("--summarize", default=None, help="이미 있는 결과 json 의 요약 표만 출력")
ap.add_argument("--n", type=int, default=1000, help="표본 수 (점검용으로 줄일 때만)")
a = ap.parse_args()


def sh(cmd, t=8):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
        return (r.stdout + r.stderr).strip()[:2000]
    except Exception as e:
        return f"(실행 못함: {e})"


def env_info(rails):
    info = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(), kernel=platform.release(), backend=a.backend,
                nvpmodel_q=sh(["nvpmodel", "-q"]), jetson_clocks_show=sh(["jetson_clocks", "--show"]),
                nv_tegra_release=(open("/etc/nv_tegra_release").readline().strip() if os.path.exists("/etc/nv_tegra_release") else None),
                power_rails=({k: list(v) for k, v in rails.items()} if rails else None), argv=sys.argv[1:])
    try:
        import tensorrt as trt; info["tensorrt"] = trt.__version__
    except Exception:
        info["tensorrt"] = None
    import glob
    for f in glob.glob("/sys/devices/platform/*/17000000.gpu/devfreq/*/cur_freq") + glob.glob("/sys/devices/platform/17000000.gpu/devfreq/*/cur_freq"):
        info["gpu_cur_freq_hz_at_start"] = open(f).read().strip()
    return info


def make_runner(name, var, prec, x):
    B = 16 if var == "b16" else 1
    if a.backend == "ort":
        return ORTRunner(os.path.join(ROOT, "models", f"{name}_{var}.onnx"), B, x), B
    return TRTRunner(os.path.join(ENGINE_DIR, f"{name}_{var}_{prec}.engine"), B, x), B


def lat_stats(ms):
    ms = np.asarray(ms)
    return dict(mean=float(ms.mean()), median=float(np.median(ms)), std=float(ms.std()), p95=float(np.percentile(ms, 95)), p99=float(np.percentile(ms, 99)), min=float(ms.min()), max=float(ms.max()), n=int(ms.size))


def power_run(r, nb, B, rails):
    s = PowerSampler(rails); s.start()
    t_start = time.perf_counter(); time.sleep(a.idle_s); t_idle_end = time.perf_counter()
    t0 = time.perf_counter(); done = 0
    while True:
        for bi in range(nb): r.run(bi)
        done += nb * B
        if time.perf_counter() - t0 >= a.min_run_s: break
    t1 = time.perf_counter(); s.stop()
    idle = s.window(t_start + 1.0, t_idle_end); proc = s.window(t0 + 1.0, t1)
    ms = (t1 - t0) / done * 1000.0
    out = dict(samples_processed=done, run_s=t1 - t0, ms_per_sample=ms, sample_interval_ms_median=float(np.median(np.diff(s.t)) * 1000), rails={})
    for k in rails:
        pi, pp = stats(idle[k]), stats(proc[k]); dyn = pp["mean"] - pi["mean"]
        out["rails"][k] = dict(idle_mW=pi, proc_mW=pp, dynamic_mW=dyn, energy_mJ_per_sample=dyn * ms / 1000.0, energy_total_mJ_per_sample=pp["mean"] * ms / 1000.0)
    return out


def run_model(name, T, prec, rails, res):
    x16, y = load_inputs(T, 1008); x1 = x16[:1000]
    if a.n < 1000: x1 = x1[:a.n]; y = y[:a.n]
    n = len(x1)
    e_logits, e_pred, e_spk = load_expected(name)
    R = res.setdefault(f"{name}|{prec}", dict(model=name, T=T, precision=prec))
    r1 = None
    if {"acc", "latency", "power"} & set(a.tasks):
        r1, _ = make_runner(name, "b1", prec, x1)
        for i in range(a.warmup): r1.run(i % n)
        lat = []
        for i in range(n):
            t = time.perf_counter(); r1.run(i); lat.append((time.perf_counter() - t) * 1000.0)
        lg = r1.output("logits")[:n]; pred = lg.argmax(1)
        R["b1"] = dict(acc=100.0 * float((pred == y).mean()), pred_agree_with_original=int((pred == e_pred[:n]).sum()), n=n,
                       logit_maxabs_vs_original=float(np.abs(lg - e_logits[:n]).max()), latency_ms=lat_stats(lat))
        if "latency" in a.tasks:
            g = [r1.run_gpu_ms(i) for i in range(n)]
            R["b1"]["gpu_exec_ms"] = lat_stats(g) if not np.isnan(g).any() else None
        print(f"  [{name} {prec}] b1 acc {R['b1']['acc']:.1f}%  원래모델과 예측 일치 {R['b1']['pred_agree_with_original']}/{n}  지연 평균 {R['b1']['latency_ms']['mean']:.3f} ms (중앙 {R['b1']['latency_ms']['median']:.3f})", flush=True)
    if "throughput" in a.tasks or ("power" in a.tasks and 16 in a.power_batches):
        r16, _ = make_runner(name, "b16", prec, x16)
        nb = len(x16) // 16
        for i in range(5): r16.run(i % nb)
        passes = []; per = []
        for _ in range(3):
            t0 = time.perf_counter()
            for bi in range(nb):
                t = time.perf_counter(); r16.run(bi); per.append((time.perf_counter() - t) * 1000.0)
            passes.append((time.perf_counter() - t0) / len(x16) * 1000.0)
        lg = r16.output("logits")[:1000]; pred = lg.argmax(1)
        R["b16"] = dict(acc=100.0 * float((pred == y[:1000]).mean()) if n == 1000 else None, pred_agree_with_original=int((pred == e_pred).sum()) if n == 1000 else None,
                        ms_per_sample_passes=passes, ms_per_sample_mean=float(np.mean(passes)), samples_per_s=float(1000.0 / np.mean(passes)), batch_ms=lat_stats(per),
                        note="1000장을 1008 로 채워 (앞 8장 되풀이) 63 배치, 3 회 통과. 정확도는 앞 1000장")
        print(f"  [{name} {prec}] b16 {R['b16']['ms_per_sample_mean']:.4f} ms/표본 ({R['b16']['samples_per_s']:.0f} 표본/s)", flush=True)
        if "power" in a.tasks and 16 in a.power_batches and rails:
            R["power_b16"] = []
            for k in range(a.power_repeats):
                pr = power_run(r16, nb, 16, rails); R["power_b16"].append(pr)
                print(f"  [{name} {prec}] 전력 b16 #{k+1}: VDD_IN 동적 {pr['rails']['VDD_IN']['dynamic_mW']:.0f} mW, {pr['ms_per_sample']:.4f} ms/표본, {pr['rails']['VDD_IN']['energy_mJ_per_sample']:.4f} mJ/표본", flush=True)
        r16.close()
    if r1 is not None and "power" in a.tasks and 1 in a.power_batches and rails:
        R["power_b1"] = []
        for k in range(a.power_repeats):
            pr = power_run(r1, n, 1, rails); R["power_b1"].append(pr)
            print(f"  [{name} {prec}] 전력 b1 #{k+1}: VDD_IN 동적 {pr['rails']['VDD_IN']['dynamic_mW']:.0f} mW, {pr['ms_per_sample']:.4f} ms/표본, {pr['rails']['VDD_IN']['energy_mJ_per_sample']:.4f} mJ/표본", flush=True)
    if r1 is not None: r1.close()
    if "spikes" in a.tasks:
        rs, _ = make_runner(name, "b1_spk", prec, x1)
        for i in range(a.warmup): rs.run(i % n)
        for i in range(n): rs.run(i)
        sp = rs.output("layer_spikes")[:n].astype(np.float64); lg = rs.output("logits")[:n]
        d = np.abs(sp - e_spk[:n])
        R["spikes"] = dict(layer_mean_trt=sp.mean(0).tolist(), layer_mean_original=e_spk[:n].mean(0).tolist(), total_mean_trt=float(sp.sum(1).mean()), total_mean_original=float(e_spk[:n].sum(1).mean()),
                           total_rel_diff=float(sp.sum(1).mean() / e_spk[:n].sum(1).mean() - 1), samples_all_layers_exact=int((d.sum(1) == 0).sum()), mean_abs_diff_per_sample=float(d.sum(1).mean()),
                           acc=100.0 * float((lg.argmax(1) == y).mean()), n=n)
        print(f"  [{name} {prec}] 층별 스파이크 합: 이미지당 {R['spikes']['total_mean_trt']:.1f} (원래 {R['spikes']['total_mean_original']:.1f}, 차 {100*R['spikes']['total_rel_diff']:+.3f}%), 전 층 정확히 같은 표본 {R['spikes']['samples_all_layers_exact']}/{n}", flush=True)
        rs.close()


def summarize(res):
    def f(v, d=2): return "-" if v is None else f"{v:.{d}f}"
    rows = []
    for key, R in res["results"].items():
        b1, b16 = R.get("b1", {}), R.get("b16", {})
        p1 = R.get("power_b1"); p16 = R.get("power_b16")
        def pw(p, rail="VDD_IN"):
            if not p: return None, None
            return float(np.mean([q["rails"][rail]["dynamic_mW"] for q in p])), float(np.mean([q["rails"][rail]["energy_mJ_per_sample"] for q in p]))
        d1, e1 = pw(p1); d16, e16 = pw(p16)
        rows.append(dict(model=R["model"], prec=R["precision"], acc=b1.get("acc"), agree=b1.get("pred_agree_with_original"), lat=(b1.get("latency_ms") or {}).get("mean"), med=(b1.get("latency_ms") or {}).get("median"),
                         gpu=((b1.get("gpu_exec_ms") or {}) or {}).get("mean"), thr=b16.get("ms_per_sample_mean"), d1=d1, e1=e1, d16=d16, e16=e16,
                         spk=(R.get("spikes") or {}).get("total_rel_diff")))
    hdr = ["model", "prec", "acc%", "일치/1000", "지연ms(b1)", "중앙", "GPUms", "b16 ms/표본", "동적mW b1", "mJ/표본 b1", "동적mW b16", "mJ/표본 b16", "스파이크합 차%"]
    print("\n" + " | ".join(hdr)); print("-" * 160)
    def line(r): return " | ".join([r["model"], r["prec"], f(r["acc"], 1), str(r["agree"] if r["agree"] is not None else "-"), f(r["lat"], 3), f(r["med"], 3), f(r["gpu"], 3), f(r["thr"], 4),
                                     f(r["d1"], 0), f(r["e1"], 3), f(r["d16"], 0), f(r["e16"], 3), "-" if r["spk"] is None else f"{100*r['spk']:+.3f}"])
    groups = {}
    for r in rows:
        print(line(r)); groups.setdefault((r["model"].rsplit("_s", 1)[0], r["prec"]), []).append(r)
    print("\n방법별 평균 (시드 평균, n = 시드 수)")
    for (g, p), rs in groups.items():
        m = lambda k: (None if all(r[k] is None for r in rs) else float(np.mean([r[k] for r in rs if r[k] is not None])))
        print(f"{g} {p} (n={len(rs)}) | acc {f(m('acc'),2)} | 지연 {f(m('lat'),3)} ms | b16 {f(m('thr'),4)} ms/표본 | 동적 b1 {f(m('d1'),0)} mW, {f(m('e1'),3)} mJ | 동적 b16 {f(m('d16'),0)} mW, {f(m('e16'),3)} mJ")


if a.summarize:
    summarize(json.load(open(a.summarize))); sys.exit(0)
man = load_manifest(); names = a.models or list(man["models"])
rails = find_power_rails() if "power" in a.tasks else None
if "power" in a.tasks and not rails:
    print("[경고] INA3221 레일 (VDD_IN, VDD_CPU_GPU_CV, VDD_SOC) 을 /sys/bus/i2c/drivers/ina3221/*/hwmon/ 에서 찾지 못해 전력은 건너뛴다.", flush=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
out = a.out or os.path.join(ROOT, "results", time.strftime("bench_%Y%m%d_%H%M%S.json"))
res = dict(env=env_info(rails), definition=dict(
    latency="배치 1, 준비 {}회 버림, 표본당 호스트 시간 (H2D + 실행 + D2H + 동기화), 입력은 미리 float32 로 pinned 메모리에 올림".format(a.warmup),
    power="동적 전력 = 처리 구간 평균 - 유휴 평균 (각 구간 맨 앞 1초 제외), 유휴 {} 초 / 처리 {} 초 이상, {}회 반복; 표본당 에너지 = 동적 전력 x 처리 표본당 시간; 전력 = 전압 x 전류 (INA3221), 간격 약 10 ms".format(a.idle_s, a.min_run_s, a.power_repeats)),
    results={})
for name in names:
    for prec in a.precisions:
        print(f"== {name} {prec}", flush=True)
        run_model(name, man["models"][name]["T"], prec, rails, res["results"])
        json.dump(res, open(out, "w"), indent=1)
print(f"\n결과 저장: {out}")
summarize(res)
