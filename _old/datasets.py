from __future__ import annotations

import re
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from typing import Optional, List, Tuple

from logging_utils import setup_logger
from config_paths import DATA_DIR


logger = setup_logger()


def _song_id_from_path(p: Path) -> Optional[int]:
    m = re.findall(r"(\d+)", p.stem)
    return int(m[0]) if m else None


def pad_and_mask(batch: List[Tuple[np.ndarray, np.ndarray]]):
    maxT = max(x.shape[0] for x, _ in batch)
    Xs, Ys, Ms = [], [], []
    for X, Y in batch:
        T = X.shape[0]
        pad = maxT - T
        if pad > 0:
            X = np.pad(X, ((0, pad), (0, 0)), mode="edge")
            Y = np.pad(Y, ((0, pad), (0, 0)), mode="edge")
            m = np.zeros((maxT,), np.float32)
            m[:T] = 1
        else:
            m = np.ones((maxT,), np.float32)
        Xs.append(torch.from_numpy(X).float())
        Ys.append(torch.from_numpy(Y).float())
        Ms.append(torch.from_numpy(m).float())
    return torch.stack(Xs, 0), torch.stack(Ys, 0), torch.stack(Ms, 0)


class DEAMDataset:
    def __init__(self, root: Path = DATA_DIR / "DEAM"):
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        ann = self.root / "annotations" / "annotations averaged per song"
        self.val_csv = ann / "dynamic (per second annotations)" / "valence.csv"
        self.aro_csv = ann / "dynamic (per second annotations)" / "arousal.csv"
        self.cache_dir = self.root / "embeddings"

    def _read_dynamic_map(self, csv_path: Path):
        df = pd.read_csv(csv_path)
        cols = [c.lower().strip() for c in df.columns]
        if {"song_id", "time", "value"} <= set(cols):
            sid = df.columns[cols.index("song_id")]
            tcol = df.columns[cols.index("time")]
            vcol = df.columns[cols.index("value")]
            by = {}
            for row in df[[sid, tcol, vcol]].itertuples(index=False):
                by.setdefault(int(getattr(row, sid)), []).append((int(getattr(row, tcol)), float(getattr(row, vcol))))
            out = {}
            for s, pairs in by.items():
                pairs.sort(key=lambda x: x[0])
                Tmax = pairs[-1][0] + 1
                arr = np.full((Tmax,), np.nan, np.float32)
                for t, v in pairs:
                    arr[t] = v
                n = np.isnan(arr)
                if n.any():
                    idx = np.where(~n)[0]
                    arr[n] = np.interp(np.where(n)[0], idx, arr[idx])
                out[int(s)] = arr
            return out

        sid = None
        for c in df.columns:
            if str(c).lower() in ("song_id", "id", "song"):
                sid = c
                break
        if sid is None:
            df = df.copy()
            df.insert(0, "song_id", df.index)
            sid = "song_id"
        df = df.set_index(sid)
        out = {}
        for s, row in df.iterrows():
            vals = row.values.astype("float32")
            if np.isnan(vals).any():
                last = np.where(~np.isnan(vals))[0]
                if len(last) > 0:
                    vals = vals[:last[-1] + 1]
                n = np.isnan(vals)
                if n.any():
                    idx = np.where(~n)[0]
                    vals[n] = np.interp(np.where(n)[0], idx, vals[idx])
            out[int(s)] = vals
        return out

    def load_dynamic_va_maps(self):
        vmap = self._read_dynamic_map(self.val_csv)
        amap = self._read_dynamic_map(self.aro_csv)
        return vmap, amap

    def build_manifest(self) -> pd.DataFrame:
        files = [p for p in self.audio_dir.rglob("*") if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")]
        vmap, amap = self.load_dynamic_va_maps()
        rows = []
        for f in files:
            sid = _song_id_from_path(f)
            rows.append({"song_id": sid, "audio_path": str(f), "has_ann": (sid in vmap and sid in amap)})
        df = pd.DataFrame(rows).dropna(subset=["song_id"])
        return df[df["has_ann"] == True].copy()

    def cache_file(self, song_id: int, extractor_name: str) -> Path:
        return self.cache_dir / extractor_name / f"{song_id}.npy"
