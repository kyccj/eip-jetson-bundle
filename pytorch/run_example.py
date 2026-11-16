"""Run one checkpoint on the bundle frames and compare with the stored reference (pred, logits).
usage: python run_example.py [--name ours_T16_s42] [--data ../data] [--n 64] [--device cpu|cuda]
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_vggsnn import load_checkpoint
from to_chip_b import to_chip_b, add_ones_channel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ours_T16_s42")
    ap.add_argument("--data", default=None, help="folder with frames_T*.npz and expected_*.npz (default: ../data or ./data)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    data = a.data or next(p for p in (os.path.join(HERE, "..", "data"), os.path.join(HERE, "data")) if os.path.isdir(p))
    man = {m["name"]: m for m in json.load(open(os.path.join(HERE, "ckpt_manifest.json")))["models"]}[a.name]
    T = man["T"]
    d = np.load(os.path.join(data, f"frames_T{T}.npz"))
    x = torch.from_numpy(d["x"][: a.n]).float().to(a.device)
    y = d["y"][: a.n]
    model = load_checkpoint(os.path.join(HERE, man["pth"]), T, a.device)
    ref = os.path.join(data, f"expected_{a.name}.npz")
    e = np.load(ref) if os.path.exists(ref) else None
    for bs in (1, 16):
        logits, spk = [], []
        for i in range(0, len(x), bs):          # state is reset inside every call, so batches are independent
            lo, sp = model(x[i:i + bs], return_spikes=True)
            logits.append(lo.cpu()); spk.append(sp.cpu())
        logits = torch.cat(logits).numpy(); pred = logits.argmax(1); spk = torch.cat(spk).numpy()
        msg = f"batch {bs:2d}: acc {100 * (pred == y).mean():.1f}%  spikes/sample {spk.sum(1).mean():.0f}"
        if e is not None:
            msg += f"  pred match {(pred == e['pred'][: a.n]).sum()}/{a.n}  max|dlogit| {np.abs(logits - e['logits'][: a.n]).max():.2e}"
            msg += f"  layer spikes equal {(spk == e['layer_spikes'][: a.n]).all()}"
        print(msg)
    cb = to_chip_b(model)                       # chip-B form: BN folded, constant 3rd input channel
    with torch.no_grad():
        lo = torch.cat([cb(add_ones_channel(x[i:i + 16])).cpu() for i in range(0, len(x), 16)]).numpy()
    print(f"chip-B form: pred same as original {(lo.argmax(1) == pred).sum()}/{a.n}")


if __name__ == "__main__":
    main()
