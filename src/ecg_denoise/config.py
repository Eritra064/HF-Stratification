import os
from dataclasses import dataclass, field
from typing import List, Tuple

import yaml


@dataclass
class Config:
    csv_dir: str = "data/csv"
    out_root: str = "outputs"
    fs: int = 500
    window_sec: float = 5.0
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    base_ch: int = 32
    num_workers: int = 2
    seed: int = 42
    use_imu: bool = True
    snr_db_range: Tuple[float, float] = (0.0, 12.0)
    eval_corruption_mode: str = "eval"
    acc_cols: List[str] = field(default_factory=lambda: ["patch_ACC_lat", "patch_ACC_hf", "patch_ACC_dv"])
    target_col: str = "ECG_lead_I"
    train_patients: List[str] = field(default_factory=list)
    val_patients: List[str] = field(default_factory=list)
    test_patients: List[str] = field(default_factory=list)

    @property
    def window(self) -> int:
        return int(self.fs * self.window_sec)

    @property
    def train_stride(self) -> int:
        return self.window // 4

    @property
    def eval_stride(self) -> int:
        return self.window

    @property
    def tag(self) -> str:
        return "imu" if self.use_imu else "noimu"

    @property
    def out_dir(self) -> str:
        return os.path.join(self.out_root, f"denoise_{self.tag}")

    @property
    def model_path(self) -> str:
        return os.path.join(self.out_dir, f"ecg_denoise_{self.tag}.pth")


def load_config(path, overrides=None) -> Config:
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    if overrides:
        d.update({k: v for k, v in overrides.items() if v is not None})
    cfg = Config(**d)
    cfg.snr_db_range = tuple(cfg.snr_db_range)
    return cfg
