from pathlib import Path
import re

import numpy as np
import pandas as pd


class DEAMDataset:
    def __init__(self, root_dir: Path, out_embeddings_dir: Path):
        self.audio_dir = root_dir / "audio"
        ann = root_dir / "annotations/annotations averaged per song"
        self.val_csv = ann / "dynamic (per second annotations)/valence.csv"
        self.aro_csv = ann / "dynamic (per second annotations)/arousal.csv"
        self.embeddings_dir = out_embeddings_dir

    @staticmethod
    def song_id_from_path(p: Path) -> int:
        m = re.findall(r"(\d+)", p.stem)
        return int(m[0])

    @staticmethod
    def _read_dynamic_map(csv_path: Path):
        df = pd.read_csv(csv_path)

        # first column is song_id, rest are values
        out = {}
        for _, row in df.iterrows():
            song_id = int(row.iloc[0])
            vals = row.iloc[1:].values.astype("float32")

            # handle NaN values with interpolation if needed
            if np.isnan(vals).any():
                n = np.isnan(vals)
                idx = np.where(~n)[0]
                if len(idx) > 1:
                    vals[n] = np.interp(np.where(n)[0], idx, vals[idx])

            out[song_id] = vals
        return out

    @property
    def va_maps(self):
        vmap = self._read_dynamic_map(self.val_csv)
        amap = self._read_dynamic_map(self.aro_csv)
        return vmap, amap
