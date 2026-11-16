"""Speck 9-layer VGGSNN (pyramid_late64) for inference with PyTorch + sinabs.

Needs torch and sinabs only (tested API: torch 2.5, sinabs 3.0/3.1).

Network (state_dict keys are the indices of nn.Sequential `net`):
  0 Conv2d(2,8,3)   1 BatchNorm2d(8)   2 IAF   3 SumPool2d(2)
  4 Conv2d(8,16,3)                     5 IAF   6 SumPool2d(2)
  7 Conv2d(16,32,3)                    8 IAF   9 SumPool2d(2)
  10 Conv2d(32,32,3,stride 2)         11 IAF
  12 Conv2d(32,64,1)                  13 IAF
  14 Conv2d(64,64,3)                  15 IAF
  16 Conv2d(64,64,3)                  17 IAF
  18 Conv2d(64,128,1)                 19 IAF
  20 Flatten  21 Linear(512,10)       22 IAF
No conv/linear bias. IAF = integrate-and-fire, threshold 1.0, lower bound of the membrane
potential -1.0, several spikes per step allowed (MultiSpike), subtract reset (MembraneSubtract).

Checkpoint file (.pth, plain dict, loadable with weights_only=True):
  "model_state_dict": weights of the keys above. Per key `<i>.weight` (Conv/Linear),
      `1.weight|bias|running_mean|running_var|num_batches_tracked` (BatchNorm),
      `<i>.min_v_mem`, `<i>.spike_threshold` (IAF scalars, -1.0 and 1.0).
      The batch-dependent membrane state `<i>.v_mem` is not stored.
  "epoch", "val_acc": training-time bookkeeping (not used for inference).

Input  x: [B, T, 2, 32, 32] event-count frames (uint8 or float). Output logits: [B, 10] (mean over T).
With return_spikes=True also returns the number of output spikes per layer [B, 9]
(conv0..conv7, fc; summed over T). Membrane state is reset at every forward call, so every call
processes independent samples (no state carries over between calls).
"""
import torch
import torch.nn as nn
import sinabs.layers as sl
from sinabs.activation import MembraneSubtract, MultiSpike

N_CLASSES = 10
IAF_IDX = (2, 5, 8, 11, 13, 15, 17, 19, 22)   # positions of the 9 IAF layers in `net`


def _iaf(T):
    return sl.IAFSqueeze(num_timesteps=T, spike_threshold=1.0, min_v_mem=-1.0,
                         spike_fn=MultiSpike, reset_fn=MembraneSubtract())


def _conv(cin, cout, k, stride=1):
    return nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False)


class SpikingVGG(nn.Module):
    def __init__(self, T, in_channels=2, bn=True):
        super().__init__()
        self.T = int(T)
        L = []
        L += [_conv(in_channels, 8, 3)] + ([nn.BatchNorm2d(8)] if bn else []) + [_iaf(T), sl.SumPool2d(2)]
        L += [_conv(8, 16, 3), _iaf(T), sl.SumPool2d(2)]
        L += [_conv(16, 32, 3), _iaf(T), sl.SumPool2d(2)]
        L += [_conv(32, 32, 3, 2), _iaf(T)]
        L += [_conv(32, 64, 1), _iaf(T)]
        L += [_conv(64, 64, 3), _iaf(T)]
        L += [_conv(64, 64, 3), _iaf(T)]
        L += [_conv(64, 128, 1), _iaf(T)]
        L += [nn.Flatten(), nn.Linear(512, N_CLASSES, bias=False), _iaf(T)]
        self.net = nn.Sequential(*L)
        self.in_channels, self.bn = in_channels, bn
        self._iafs = [m for m in self.net if isinstance(m, sl.IAFSqueeze)]

    def reset_states(self):
        for m in self._iafs:
            m.reset_states()

    @torch.no_grad()
    def forward(self, x, return_spikes=False):
        B, T = x.shape[:2]
        assert T == self.T, f"model built for T={self.T}, input has T={T}"
        self.reset_states()
        counts = []
        hooks = []
        if return_spikes:
            hooks = [m.register_forward_hook(
                lambda mod, i, o: counts.append(o.reshape(B, -1).sum(1).to(torch.int64))) for m in self._iafs]
        try:
            h = x.reshape(B * T, *x.shape[2:]).to(torch.float32)
            out = self.net(h).reshape(B, T, -1).mean(1)
        finally:
            for hk in hooks:
                hk.remove()
        return (out, torch.stack(counts, 1)) if return_spikes else out


def build_model(T, in_channels=2, bn=True, device="cpu"):
    """Eval-mode network. in_channels=3, bn=False is the chip-B form (see to_chip_b.py)."""
    return SpikingVGG(T, in_channels, bn).to(device).eval()


def load_checkpoint(path, T, device="cpu"):
    """Load a bundle .pth into a fresh model; raises if any key is missing or unexpected."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in ck["model_state_dict"].items() if not k.endswith(".v_mem")}
    model = build_model(T)
    res = model.net.load_state_dict(sd, strict=False)   # v_mem (batch-dependent state) is not stored
    missing = [k for k in res.missing_keys if not k.endswith(".v_mem")]
    assert not missing and not res.unexpected_keys, (missing, res.unexpected_keys)
    return model.to(device).eval()
