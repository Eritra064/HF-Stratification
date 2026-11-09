import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .artifacts import make_artifact, window_rng


def patient_of(rec):
    return rec.split("-")[0]


def load_split_records(cfg):
    paths = sorted(glob.glob(os.path.join(cfg.csv_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSVs in '{cfg.csv_dir}'.")
    buckets = {"train": [], "val": [], "test": []}
    for p in paths:
        rec = os.path.splitext(os.path.basename(p))[0]
        pat = patient_of(rec)
        if pat in cfg.train_patients:
            split = "train"
        elif pat in cfg.val_patients:
            split = "val"
        elif pat in cfg.test_patients:
            split = "test"
        else:
            continue
        df = pd.read_csv(p, usecols=cfg.acc_cols + [cfg.target_col])
        acc = df[cfg.acc_cols].to_numpy(np.float32)
        clean = df[cfg.target_col].to_numpy(np.float32)
        buckets[split].append((rec, acc, clean))
    for s in buckets:
        recs = [r for r, _, _ in buckets[s]]
        mins = sum(len(c) for _, _, c in buckets[s]) / cfg.fs / 60
        print(f"  {s:5s}: {len(buckets[s])} records {recs}  ({mins:.1f} min)")
    return buckets


def normalize(buckets):
    """Per-record z-scoring of ACC; one global target scaler fit on train Lead I."""
    y_scaler = StandardScaler()
    y_scaler.fit(np.concatenate([c for _, _, c in buckets["train"]]).reshape(-1, 1))
    print(f"\n  target scaler (train Lead I): mean={y_scaler.mean_[0]:.4f} "
          f"std={y_scaler.scale_[0]:.4f} mV")
    out = {}
    for split, recs in buckets.items():
        norm = []
        for rec, acc, clean in recs:
            acc_n = StandardScaler().fit_transform(acc).astype(np.float32)  # per-record
            norm.append((rec, acc_n, clean.astype(np.float32)))             # clean stays mV
        out[split] = norm
    return out, y_scaler


def build_input(noisy_w, acc_w, ym, ys, use_imu):
    """Stack normalized noisy ECG with ACC (or zeros) into a (4, W) array."""
    noisy_n = (noisy_w - ym) / ys
    acc_in = acc_w.T if use_imu else np.zeros_like(acc_w.T)
    return np.vstack([noisy_n[None, :], acc_in]).astype(np.float32)


class DenoiseWindows(Dataset):
    def __init__(self, records, y_scaler, cfg, stride, mode, deterministic):
        self.records = records
        self.cfg = cfg
        self.ym, self.ys = float(y_scaler.mean_[0]), float(y_scaler.scale_[0])
        self.mode = mode
        self.deterministic = deterministic
        self.index = []
        for ri, (_, acc, _) in enumerate(records):
            for s in range(0, len(acc) - cfg.window, stride):
                self.index.append((ri, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ri, s = self.index[i]
        rec, acc, clean = self.records[ri]
        if self.deterministic:
            rng = window_rng(rec, s)
        else:
            # drawn from torch's RNG so training noise is repeatable given the seed
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        W = self.cfg.window
        clean_w = clean[s:s + W]
        acc_w = acc[s:s + W, :]
        snr = rng.uniform(*self.cfg.snr_db_range)
        art = make_artifact(clean_w, acc_w, rng, snr, self.mode, self.cfg.fs)
        x = build_input(clean_w + art, acc_w, self.ym, self.ys, self.cfg.use_imu)
        clean_n = ((clean_w - self.ym) / self.ys).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(clean_n)
