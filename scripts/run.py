import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_denoise.config import load_config
from ecg_denoise.data import DenoiseWindows, load_split_records, normalize
from ecg_denoise.evaluate import evaluate
from ecg_denoise.model import DenoiseNet
from ecg_denoise.train import train_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--csv-dir", dest="csv_dir")
    p.add_argument("--out-root", dest="out_root")
    p.add_argument("--epochs", type=int)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--imu", dest="use_imu", action="store_true")
    g.add_argument("--no-imu", dest="use_imu", action="store_false")
    p.set_defaults(use_imu=None)
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}   use_imu={cfg.use_imu}   eval_mode={cfg.eval_corruption_mode}")

    print("\nLoading records")
    buckets = load_split_records(cfg)
    print("\nNormalizing (per-record ACC, global target on train)")
    norm, y_scaler = normalize(buckets)

    tr = DenoiseWindows(norm["train"], y_scaler, cfg, cfg.train_stride, "train", deterministic=False)
    va = DenoiseWindows(norm["val"], y_scaler, cfg, cfg.eval_stride, cfg.eval_corruption_mode, deterministic=True)
    print(f"\n  train windows: {len(tr):,}   val windows: {len(va):,}")
    tr_dl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    va_dl = DataLoader(va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = DenoiseNet(in_ch=4, bc=cfg.base_ch).to(device)
    print(f"  model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining")
    train_model(model, tr_dl, va_dl, cfg, device)

    print("\nEvaluating best checkpoint on held-out TEST patients")
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    evaluate(model, norm["test"], y_scaler, cfg, device)
    print("\nDone")


if __name__ == "__main__":
    main()
