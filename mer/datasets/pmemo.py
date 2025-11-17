from pathlib import Path
import pandas as pd
import numpy as np
import re

class PMEmoDataset:
    def __init__(self, root_dir: Path, out_embeddings_dir: Path):
        self.audio_dir = root_dir / "chorus"
        self.annotations_file = root_dir / "annotations" / "dynamic_annotations.csv"
        self.embeddings_dir = out_embeddings_dir

    @staticmethod
    def song_id_from_path(p: Path) -> int:
        m = re.findall(r"(\d+)", p.stem)
        return int(m[0])

    def _read_va_map(self):
        df = pd.read_csv(self.annotations_file)
        v_map = {}
        a_map = {}

        for song_id, group in df.groupby("musicId"):
            vals_aro = group["Arousal(mean)"].values.astype("float32")
            vals_val = group["Valence(mean)"].values.astype("float32")

            # handle NaN values with interpolation if needed
            for arr in [vals_aro, vals_val]:
                if np.isnan(arr).any():
                    n = np.isnan(arr)
                    idx = np.where(~n)[0]
                    if len(idx) > 1:
                        arr[n] = np.interp(np.where(n)[0], idx, arr[idx])

            # convert values from [0, 1] to [-1, 1]
            vals_aro = vals_aro * 2.0 - 1.0
            vals_val = vals_val * 2.0 - 1.0

            a_map[int(song_id)] = vals_aro
            v_map[int(song_id)] = vals_val

        return v_map, a_map

    @property
    def va_maps(self):
        return self._read_va_map()
