from __future__ import annotations

import numpy as np


def _a(x):
    return np.asarray(x, dtype=float).ravel()


def pearson(y, p):
    y = _a(y)
    p = _a(p)
    if y.size < 2:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def rmse(y, p):
    y = _a(y)
    p = _a(p)
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    y = _a(y)
    p = _a(p)
    if y.size < 2:
        return float("nan")
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def ccc(y, p):
    y = _a(y)
    p = _a(p)
    if y.size < 2:
        return float("nan")
    my, mp = np.mean(y), np.mean(p)
    vy, vp = np.var(y), np.var(p)
    cov = np.mean((y - my) * (p - mp))
    denom = vy + vp + (my - mp) ** 2
    if denom == 0:
        return float("nan")
    return float(2 * cov / denom)


def metrics_dict(y, p):
    return {
        "CCC": ccc(y, p),
        "Pearson": pearson(y, p),
        "R2": r2(y, p),
        "RMSE": rmse(y, p),
    }


def labels_convert(y, src: str, dst: str):
    y = np.asarray(y, dtype=float)
    if src == dst:
        return y
    if src == "19" and dst == "norm":
        return (y - 5.0) / 4.0
    if src == "norm" and dst == "19":
        return 4.0 * y + 5.0
    raise ValueError(f"Unsupported scale conversion {src} -> {dst}")
