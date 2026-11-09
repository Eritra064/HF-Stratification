import argparse
import time

import torch
import torch.nn as nn

from ecg_denoise.config import load_config
from ecg_denoise.model import DenoiseNet


def count_macs(model, x):
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0]

    def tconv_hook(m, inp, out):
        total[0] += inp[0].numel() // inp[0].shape[0] * (m.out_channels // m.groups) * m.kernel_size[0]

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose1d):
            hooks.append(m.register_forward_hook(tconv_hook))
        elif isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return total[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--runs", type=int, default=50)
    args = p.parse_args()
    cfg = load_config(args.config)

    torch.set_num_threads(1)
    model = DenoiseNet(in_ch=4, bc=cfg.base_ch).eval()
    x = torch.randn(1, 4, cfg.window)

    params = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, x)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t0 = time.perf_counter()
        for _ in range(args.runs):
            model(x)
        ms = (time.perf_counter() - t0) / args.runs * 1000

    print(f"Window:            {cfg.window_sec:g} s ({cfg.window} samples @ {cfg.fs} Hz)")
    print(f"Parameters:        {params:,}  ({params * 4 / 1024:.0f} KB at float32)")
    print(f"Conv MACs/window:  {macs / 1e6:.1f} M")
    print(f"CPU latency:       {ms:.1f} ms per window (1 thread)")


if __name__ == "__main__":
    main()
