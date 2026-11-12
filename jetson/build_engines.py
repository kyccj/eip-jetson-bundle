#!/usr/bin/env python3
"""ONNX -> TensorRT 엔진 (FP32 / FP16). numpy + tensorrt 만 필요. sudo 불필요.

예)  python3 jetson/build_engines.py                       # 12개 모델 x (b1, b16, b1_spk) x (fp32, fp16)  (오래 걸린다)
     python3 jetson/build_engines.py --models ours_T16_s42 base_T8_s42 sip2e8_T16_s42 --precisions fp32 fp16 --variants b1 b16
엔진은 engines/<모델>_<변형>_<정밀도>.engine. 이미 있으면 건너뛴다 (--force 로 덮어쓰기). 타이밍 캐시 engines/timing.cache 를 공유해 뒤쪽 빌드가 빨라진다.
FP32 엔진은 TF32 를 끈다 (진짜 FP32). FP16 엔진은 FP16 + FP32 혼합을 TensorRT 가 고른다.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eipcommon import ROOT, ENGINE_DIR, load_manifest
import tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=None, help="기본: manifest 의 전부")
ap.add_argument("--variants", nargs="*", default=["b1", "b16", "b1_spk"], choices=["b1", "b16", "b1_spk"])
ap.add_argument("--precisions", nargs="*", default=["fp32", "fp16"], choices=["fp32", "fp16"])
ap.add_argument("--workspace_mb", type=int, default=1024)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
man = load_manifest(); names = a.models or list(man["models"])
os.makedirs(ENGINE_DIR, exist_ok=True)
logger = trt.Logger(trt.Logger.WARNING)
tc_path = os.path.join(ENGINE_DIR, "timing.cache")
tc_bytes = open(tc_path, "rb").read() if os.path.exists(tc_path) else b""
print(f"TensorRT {trt.__version__}  모델 {len(names)}  변형 {a.variants}  정밀도 {a.precisions}", flush=True)
for name in names:
    for var in a.variants:
        for prec in a.precisions:
            out = os.path.join(ENGINE_DIR, f"{name}_{var}_{prec}.engine")
            if os.path.exists(out) and not a.force:
                print("건너뜀", out); continue
            onnx_path = os.path.join(ROOT, "models", f"{name}_{var}.onnx")
            t0 = time.time()
            builder = trt.Builder(logger); net = builder.create_network(); parser = trt.OnnxParser(net, logger)
            if not parser.parse_from_file(onnx_path):
                for i in range(parser.num_errors): print(parser.get_error(i))
                sys.exit(f"ONNX 읽기 실패: {onnx_path}")
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace_mb << 20)
            cfg.clear_flag(trt.BuilderFlag.TF32)
            if prec == "fp16": cfg.set_flag(trt.BuilderFlag.FP16)
            cache = cfg.create_timing_cache(tc_bytes); cfg.set_timing_cache(cache, False)
            ser = builder.build_serialized_network(net, cfg)
            if ser is None: sys.exit(f"엔진 빌드 실패: {out}")
            open(out, "wb").write(bytes(ser))
            tc_bytes = bytes(cache.serialize()); open(tc_path, "wb").write(tc_bytes)
            print(f"{os.path.basename(out)}  {os.path.getsize(out)/1e6:.2f} MB  {time.time()-t0:.0f}s", flush=True)
