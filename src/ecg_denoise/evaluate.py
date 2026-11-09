import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from .artifacts import make_artifact, window_rng
from .data import build_input, patient_of


def evaluate(model, test_records, y_scaler, cfg, device):
    ym, ys = float(y_scaler.mean_[0]), float(y_scaler.scale_[0])
    W = cfg.window
    model.eval()
    by_pat = {}
    for rec, acc, clean in test_records:
        by_pat.setdefault(patient_of(rec), []).append((rec, acc, clean))
    pred_dir = os.path.join(cfg.out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    rows = []
    for pat, recs in sorted(by_pat.items()):
        cl, no, de = [], [], []
        with torch.no_grad():
            for rec, acc, clean in recs:
                for s in range(0, len(acc) - W, cfg.eval_stride):
                    rng = window_rng(rec, s)
                    cw = clean[s:s + W]; aw = acc[s:s + W, :]
                    snr = rng.uniform(*cfg.snr_db_range)
                    art = make_artifact(cw, aw, rng, snr, cfg.eval_corruption_mode, cfg.fs)
                    noisy = cw + art
                    x = build_input(noisy, aw, ym, ys, cfg.use_imu)
                    out_n = model(torch.from_numpy(x[None]).to(device)).cpu().numpy()[0, 0]
                    cl.append(cw); no.append(noisy); de.append(out_n * ys + ym)
        cl = np.concatenate(cl); no = np.concatenate(no); de = np.concatenate(de)
        in_rmse = float(np.sqrt(mean_squared_error(cl, no)))
        out_rmse = float(np.sqrt(mean_squared_error(cl, de)))
        in_r = float(np.corrcoef(cl, no)[0, 1])
        out_r = float(np.corrcoef(cl, de)[0, 1])
        ss_res = float(np.sum((de - cl) ** 2)); ss_tot = float(np.sum((cl - cl.mean()) ** 2))
        out_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append(dict(patient=pat, in_rmse_mV=in_rmse, out_rmse_mV=out_rmse,
                         in_r=in_r, out_r=out_r, out_r2=out_r2, n=len(cl)))
        pd.DataFrame({"ECG_lead_I_true_mV": cl, "ECG_lead_I_noisy_mV": no,
                      "ECG_lead_I_denoised_mV": de}
                     ).to_csv(os.path.join(pred_dir, f"{pat}_denoise.csv"), index=False)
        print(f"  {pat}:  in(r={in_r:.3f}, RMSE={in_rmse:.4f})  ->  "
              f"out(r={out_r:.3f}, RMSE={out_rmse:.4f}, R2={out_r2:.3f})")
    df = pd.DataFrame(rows)
    mean = {k: (df[k].mean() if k != "patient" else "MEAN") for k in df.columns}
    mean["n"] = int(df["n"].sum())
    df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    df.to_csv(os.path.join(cfg.out_dir, f"metrics_denoise_{cfg.tag}.csv"), index=False)
    print("\n" + df.to_string(index=False))
    return df
