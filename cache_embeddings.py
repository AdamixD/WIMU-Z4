from __future__ import annotations

import argparse
import csv
import numpy as np

from pathlib import Path
from tqdm import tqdm

from config_paths import RESULTS_DIR
from datasets import DEAMDataset
from embeddings import load_audio_mono, LibrosaMFCCChroma
from logging_utils import setup_logger


logger = setup_logger()


def main():
    ap = argparse.ArgumentParser(description="Cache embeddings for DEAM")
    ap.add_argument("--model-name", type=str, default="shared_cache")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    DATASET_NAME = "DEAM"
    report_dir = RESULTS_DIR / DATASET_NAME / args.model_name
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = DEAMDataset()
    extractor = LibrosaMFCCChroma()
    df = dataset.build_manifest()
    pairs = [(int(r.song_id), Path(r.audio_path)) for r in df.itertuples(index=False)]
    if args.limit: 
        pairs = pairs[:args.limit]

    failures = []
    for song_id, audio_path in tqdm(pairs, desc=f"Caching"):
        cfile = dataset.cache_file(song_id, "librosa")
        if cfile.exists(): 
            continue
        try:
            y = load_audio_mono(audio_path, sr=16000)
            X = extractor(y, 16000)
            cfile.parent.mkdir(parents=True, exist_ok=True)
            np.save(cfile, X.astype("float32"))
        except Exception as e:
            failures.append((str(audio_path), str(e)))

    if failures:
        out = report_dir / f"cache_failures.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["audio_path", "reason"])
            w.writerows(failures)
        logger.warning(f"Wrote failures report: {out}")
    logger.info("Caching completed.")


if __name__ == "__main__":
    main()
