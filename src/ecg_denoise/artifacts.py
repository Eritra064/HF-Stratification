import zlib

import numpy as np
from scipy.signal import butter, filtfilt


def window_rng(rec, start):
    return np.random.default_rng(zlib.crc32(f"{rec}_{start}".encode()))


def _bandlimited(W, rng, lo, hi, fs):
    x = rng.standard_normal(W)
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x).astype(np.float32)


def _envelope(W, rng, fs, n=3):
    t = np.arange(W) / fs
    e = np.zeros(W, np.float32)
    for _ in range(n):
        f = rng.uniform(0.05, 0.5)
        e += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    e -= e.min()
    return (e / (np.ptp(e) + 1e-8)).astype(np.float32)


def make_artifact(clean_win, acc_win, rng, snr_db, mode, fs):

    W = len(clean_win)
    # random linear mix of the 3 ACC axes (weights unknown to the model)
    w = rng.standard_normal(3)
    m = acc_win @ w
    # random short FIR kernel: unknown coupling dynamics
    k = rng.standard_normal(int(rng.integers(8, 24)))
    k /= (np.abs(k).sum() + 1e-8)
    m = np.convolve(m, k, mode="same")
    sd = np.std(m) + 1e-8

    if mode == "train":
        m_nl = m + 0.5 * np.sign(m) * (m ** 2) / sd
    else:
        m_nl = np.tanh(1.5 * m / sd) * sd
    acc_art = _envelope(W, rng, fs) * m_nl
    acc_art /= (np.std(acc_art) + 1e-8)

    # IMU-independent noise: baseline wander + high-frequency noise
    bw = _bandlimited(W, rng, 0.05, 0.5, fs)
    hf = _bandlimited(W, rng, 20.0, min(120.0, fs / 2 - 1), fs)
    indep = rng.uniform(0.5, 1.0) * bw / (np.std(bw) + 1e-8) \
          + rng.uniform(0.3, 0.8) * hf / (np.std(hf) + 1e-8)
    art = acc_art + rng.uniform(0.5, 1.2) * indep

    # scale to target SNR
    sig_p = np.var(clean_win) + 1e-12
    art *= np.sqrt((sig_p / (10 ** (snr_db / 10))) / (np.var(art) + 1e-12))
    return art.astype(np.float32)
