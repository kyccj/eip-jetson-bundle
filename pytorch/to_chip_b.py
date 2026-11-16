"""Chip-B form of the network: conv0 + BatchNorm folded into conv0 with one extra constant input channel.

y = BN(W*x + 0) = W' * x + b  with  W' = W * gamma / sqrt(var + eps),  b = beta - gamma * mean / sqrt(var + eps).
The bias b is the centre tap (padding 1) of a third input channel that is 1.0 at every pixel and every time step,
so every output pixel gets +b at every step. Layers 1..8 are unchanged (no BatchNorm there).
Floating point only: the 8-bit weight discretization of the chip is not applied.
"""
import torch
from model_vggsnn import build_model


@torch.no_grad()
def to_chip_b(model):
    bn, conv = model.net[1], model.net[0]
    assert isinstance(bn, torch.nn.BatchNorm2d) and conv.in_channels == 2
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale.view(-1, 1, 1, 1)
    b = bn.bias - scale * bn.running_mean
    new = build_model(model.T, in_channels=3, bn=False)
    sd = {}
    for k, v in model.net.state_dict().items():
        i = int(k.split(".")[0])
        if i == 0:
            w3 = torch.zeros(w.shape[0], 3, *w.shape[2:])
            w3[:, :2] = w
            w3[:, 2, 1, 1] = b
            sd["0.weight"] = w3
        elif i == 1 or k.endswith(".v_mem"):
            continue
        else:
            sd[str(i - 1) + k[len(str(i)):]] = v    # BatchNorm removed: later indices shift by one
    res = new.net.load_state_dict(sd, strict=False)
    assert not [k for k in res.missing_keys if not k.endswith(".v_mem")] and not res.unexpected_keys, res
    return new.to(next(model.parameters()).device).eval()


def add_ones_channel(x):
    """[B, T, 2, H, W] -> [B, T, 3, H, W] with channel 2 = 1.0 (the constant bias carrier)."""
    x = x.to(torch.float32)
    return torch.cat([x, torch.ones_like(x[:, :, :1])], dim=2)
