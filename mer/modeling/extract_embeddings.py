import re
from pathlib import Path
from typing import Annotated, Literal

import librosa
from loguru import logger
from tqdm import tqdm
import typer
import numpy as np
import pandas as pd

from mer.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SAMPLE_RATE
from mer.extractor import LibrosaMFCCChroma


app = typer.Typer()


class DEAMDataset:
    def __init__(self, root_dir: Path, out_embeddings_dir: Path):
        self.audio_dir = root_dir / 'audio'
        ann = root_dir / 'annotations/annotations averaged per song'
        self.val_csv = ann / 'dynamic (per second annotations)/valence.csv'
        self.aro_csv = ann / 'dynamic (per second annotations)/arousal.csv'
        self.embeddings_dir = out_embeddings_dir

    @staticmethod
    def song_id_from_path(p: Path) -> int:
        m = re.findall(r'(\d+)', p.stem)
        return int(m[0])

    @staticmethod
    def _read_dynamic_map(csv_path: Path):
        df = pd.read_csv(csv_path)

        # first column is song_id, rest are values
        out = {}
        for _, row in df.iterrows():
            song_id = int(row.iloc[0])
            vals = row.iloc[1:].values.astype('float32')

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


def load_audio_mono(path, sr=None):
    """Loads an audio file as a floating point time series."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def build_manifest(dataset) -> pd.DataFrame:
    files = [
        p for p in dataset.audio_dir.rglob('*')
        if p.suffix.lower() in ('.mp3', '.wav', '.flac', '.m4a')
    ]
    vmap, amap = dataset.va_maps
    rows = []
    for f in files:
        sid = dataset.song_id_from_path(f)
        rows.append({
            'song_id': sid,
            'audio_path': str(f),
            'annotated': (sid in vmap and sid in amap)
        })
    df = pd.DataFrame(rows).dropna(subset=['song_id'])
    return df[df['annotated'] == True].copy()


@app.command()
def main(
    dataset_name: Annotated[Literal['DEAM',], typer.Option(case_sensitive=False)] = 'DEAM',
    limit: int | None = None
):
    extractor = LibrosaMFCCChroma()
    if dataset_name == 'DEAM':
        dataset = DEAMDataset(
            root_dir=RAW_DATA_DIR/dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR/dataset_name/'embeddings'
        )
    else:
        raise NotImplemented(dataset_name)

    manifest = build_manifest(dataset)[:limit]
    if limit:
        manifest = manifest[:limit]
    manifest.to_csv(dataset.embeddings_dir.parent/'manifest.csv', index=False)

    pairs = [(r.song_id, Path(r.audio_path)) for r in
             manifest.itertuples(index=False)]

    logger.info(f'Processing {dataset_name} dataset...')
    for song_id, audio_path in (pbar := tqdm(pairs, desc=f'Extracting...')):
        pbar.set_postfix_str(f'Song ID: {song_id}')

        song_embds_path = dataset.embeddings_dir/str(song_id)
        if song_embds_path.exists():
            logger.warning(f'Embeddings already exists at {song_embds_path}')
            continue

        # Essentia example used mono conversion, but does it have to be mono?
        y = load_audio_mono(audio_path, sr=SAMPLE_RATE)
        song_embds = extractor(y, SAMPLE_RATE)
        song_embds_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(song_embds_path, song_embds.astype('float32'))
    logger.success('Processing dataset complete.')


if __name__ == '__main__':
    app()
