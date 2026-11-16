# PyTorch + sinabs version of the Speck 9-layer VGGSNN

Run the same 20 checkpoints directly in PyTorch (no ONNX / TensorRT). Needs `torch` and `sinabs` only.
Put this folder inside the bundle repository (e.g. `pytorch/`) so `../data` holds `frames_T*.npz` and `expected_*.npz`, or pass `--data`.

- `ckpt/<name>.pth`: 16 models of `manifest.json` plus `sip1e8_T16_s42..45` (SIP lambda 1e-8, T=16). `ckpt_manifest.json` lists method, T, seed, source file (relative path, size, sha256) and the sha256 of each `.pth`.
  The `.pth` files hold the same weights as the originals; only the batch-dependent membrane state and the training arguments were dropped (plain dict, `torch.load(..., weights_only=True)`).
- `model_vggsnn.py`: `build_model(T)` and `load_checkpoint(path, T)`; the key layout is described at the top of the file. Input `[B, T, 2, 32, 32]`, output logits `[B, 10]`; `return_spikes=True` also gives per-layer spike counts `[B, 9]`. Membrane state is reset at every call.
- `to_chip_b.py`: `to_chip_b(model)` folds conv0 BatchNorm into a 3-input-channel conv0 (constant channel carries the bias); feed `add_ones_channel(x)`. Floating point, no 8-bit discretization.
- `run_example.py`: `python run_example.py --name base_T8_s42 --data ../data --device cuda` compares pred / logits / layer spikes with `expected_<name>.npz` at batch 1 and 16 and checks the chip-B form.
- Reference check (CPU, torch 2.5.1, sinabs 3.1.3): predictions, logits and layer spikes of all 16 models equal `expected_*.npz` exactly on the first 64 samples; chip-B form changes at most a few logits slightly (float rounding of the folded weights).
- `sip1e8_T16_s4x` have no `expected_*.npz`; their 1000-sample accuracy is 59.5 / 58.6 / 58.9 / 58.8 % (s42..s45).
- Models are small (T up to 16, 32x32); on GPU a larger batch is faster, but results do not depend on the batch size.
