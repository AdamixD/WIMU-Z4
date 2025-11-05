import re
from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
from tqdm import tqdm
import typer
import numpy as np
import pandas as pd

from mer.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from mer.modeling.embeddings import LibrosaMFCCChroma, load_audio_mono

app = typer.Typer()
SAMPLE_RATE = 16000


class DEAMDataset:
    def __init__(self, root_dir: Path, out_embeddings_dir: Path):
        self.audio_dir = root_dir / "audio"
        ann = root_dir / "annotations" / "annotations averaged per song"
        self.val_csv = ann / "dynamic (per second annotations)" / "valence.csv"
        self.aro_csv = ann / "dynamic (per second annotations)" / "arousal.csv"
        self.embeddings_dir = out_embeddings_dir

    @staticmethod
    def song_id_from_path(p: Path) -> int:
        m = re.findall(r"(\d+)", p.stem)
        return int(m[0])

    @staticmethod
    def _read_dynamic_map(csv_path: Path):
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

    @property
    def va_maps(self):
        vmap = self._read_dynamic_map(self.val_csv)
        amap = self._read_dynamic_map(self.aro_csv)
        return vmap, amap


def build_manifest(dataset) -> pd.DataFrame:
    files = [
        p for p in dataset.audio_dir.rglob("*")
        if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")
    ]
    vmap, amap = dataset.va_maps
    rows = []
    for f in files:
        sid = dataset.song_id_from_path(f)
        rows.append({"song_id": sid, "audio_path": str(f), "has_ann": (sid in vmap and sid in amap)})  # is that check right? doesn't it go through values as well?
    df = pd.DataFrame(rows).dropna(subset=["song_id"])
    return df[df["has_ann"] == True].copy()


@app.command()
def main(
    dataset_name: Annotated[Literal["deam",], typer.Option()] = "deam",
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    limit: int | None = None
):
    extractor = LibrosaMFCCChroma()
    if dataset_name == "deam":
        dataset = DEAMDataset(
            root_dir=input_path/dataset_name,
            out_embeddings_dir=output_path/dataset_name
        )
    else:
        raise NotImplemented(dataset_name)

    manifest = build_manifest(dataset)
    pairs = [(r.song_id, Path(r.audio_path)) for r in
             manifest.itertuples(index=False)]

    if limit:
        pairs = pairs[:limit]

    logger.info(f"Processing {dataset_name} dataset...")
    for song_id, audio_path in (pbar := tqdm(pairs, desc=f"Extracting...")):
        pbar.set_postfix_str(f"Song ID: {song_id}")

        song_embds_path = dataset.embeddings_dir/str(song_id)
        if song_embds_path.exists():
            logger.warning(f'Embeddings already exists at {song_embds_path}')
            continue

        y = load_audio_mono(audio_path, sr=SAMPLE_RATE)
        song_embds = extractor(y, SAMPLE_RATE)
        song_embds_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(song_embds_path, song_embds.astype("float32"))
    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
