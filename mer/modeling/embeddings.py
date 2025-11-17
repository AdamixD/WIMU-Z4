from pathlib import Path
from typing import Annotated, Literal

import librosa
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from mer.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SAMPLE_RATE
from mer.datasets.deam import DEAMDataset
from mer.datasets.pmemo import PMEmoDataset
from mer.extractor import LibrosaMFCCChroma

app = typer.Typer()


def load_audio_mono(path, sr=None):
    """Loads an audio file as a floating point time series."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def extract_embeddings(audio_path, sr=SAMPLE_RATE, extractor=None):
    if not extractor:
        extractor = LibrosaMFCCChroma()
    y = load_audio_mono(audio_path, sr=sr)
    return extractor(y, sr)


def build_manifest(dataset) -> pd.DataFrame:
    files = [
        p
        for p in dataset.audio_dir.rglob("*")
        if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")
    ]
    vmap, amap = dataset.va_maps
    rows = []
    for f in files:
        sid = dataset.song_id_from_path(f)
        rows.append(
            {
                "song_id": sid,
                "audio_path": str(f),
                "annotated": (sid in vmap and sid in amap),
                "embeddings_path": None,
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["song_id"])
    return df[df["annotated"]].copy()


@app.command()
def main(
    dataset_name: Annotated[Literal["DEAM", "PMEmo"], typer.Option(case_sensitive=False)] = "DEAM",
    limit: int | None = None,
):
    extractor = LibrosaMFCCChroma()
    if dataset_name == "DEAM":
        dataset = DEAMDataset(
            root_dir=RAW_DATA_DIR / dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR / dataset_name / "embeddings",
        )
    elif dataset_name == "PMEmo":
        dataset = PMEmoDataset(
            root_dir=RAW_DATA_DIR / dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR / dataset_name / "embeddings",
        )
    else:
        raise NotImplementedError(dataset_name)

    manifest = build_manifest(dataset)
    if limit:
        manifest = manifest[:limit]

    pairs = [(r.song_id, Path(r.audio_path)) for r in manifest.itertuples(index=False)]

    logger.info(f"Processing {dataset_name} dataset...")
    for song_id, audio_path in (pbar := tqdm(pairs, desc="Extracting...")):
        pbar.set_postfix_str(f"Song ID: {song_id}")

        song_embds_path = dataset.embeddings_dir / f"{song_id}.npy"
        manifest.loc[manifest["song_id"] == song_id, "embeddings_path"] = str(song_embds_path)

        if song_embds_path.exists():
            logger.warning(f"Embeddings already exists at {song_embds_path}")
            continue

        # Essentia example used mono conversion, but does it have to be mono?
        song_embds = extract_embeddings(audio_path, extractor=extractor)
        song_embds_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(song_embds_path, song_embds.astype("float32"))
    logger.success("Processing dataset complete.")
    manifest.to_csv(dataset.embeddings_dir.parent / "manifest.csv", index=False)


if __name__ == "__main__":
    app()
