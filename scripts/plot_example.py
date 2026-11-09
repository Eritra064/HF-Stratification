import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_denoise.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--patient", default="TRM107")
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--dur-sec", type=float, default=5.0)
    p.add_argument("--out", default="figures/example_denoising.png")
    args = p.parse_args()
    cfg = load_config(args.config)

    df = pd.read_csv(os.path.join(cfg.out_dir, "predictions", f"{args.patient}_denoise.csv"))
    i0 = int(args.start_sec * cfg.fs)
    i1 = i0 + int(args.dur_sec * cfg.fs)
    seg = df.iloc[i0:i1]
    t = np.arange(len(seg)) / cfg.fs

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, seg["ECG_lead_I_noisy_mV"], lw=0.8, color="0.6", label="Corrupted input")
    axes[0].plot(t, seg["ECG_lead_I_true_mV"], lw=0.9, color="k", label="Clean Lead I")
    axes[1].plot(t, seg["ECG_lead_I_true_mV"], lw=0.9, color="k", label="Clean Lead I")
    axes[1].plot(t, seg["ECG_lead_I_denoised_mV"], lw=0.9, color="tab:red", ls="--", label="Denoised")
    for ax in axes:
        ax.set_ylabel("mV")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(f"{args.patient} ({cfg.tag}), held-out test patient")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
