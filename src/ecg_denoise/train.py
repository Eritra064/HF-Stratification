import numpy as np
import torch
import torch.nn as nn


def _loss(pr, yb, mse, mae):
    L = min(pr.shape[2], yb.shape[2])
    return mse(pr[:, :, :L], yb[:, :, :L]) + 0.5 * mae(pr[:, :, :L], yb[:, :, :L])


def train_model(model, tr, va, cfg, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    mse, mae = nn.MSELoss(), nn.L1Loss()
    best = float("inf")
    for ep in range(1, cfg.epochs + 1):
        model.train(); tl = []
        for xb, yb in tr:
            xb = xb.to(device); yb = yb.to(device).unsqueeze(1)
            loss = _loss(model(xb), yb, mse, mae)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl.append(loss.item())
        model.eval(); vl = []
        with torch.no_grad():
            for xb, yb in va:
                xb = xb.to(device); yb = yb.to(device).unsqueeze(1)
                vl.append(_loss(model(xb), yb, mse, mae).item())
        t, v = float(np.mean(tl)), float(np.mean(vl))
        sch.step(v)
        if v < best:
            best = v; torch.save(model.state_dict(), cfg.model_path)
        if ep % 5 == 0 or ep == 1:
            print(f"  Epoch {ep:03d}/{cfg.epochs}  train={t:.4f}  val={v:.4f}")
    print(f"  Best val={best:.4f}  (saved {cfg.model_path})")
