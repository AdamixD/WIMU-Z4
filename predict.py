from __future__ import annotations

from pathlib import Path

import argparse
import json
import numpy as np
import pandas as pd
import essentia.standard as es
import torch

from logging_utils import setup_logger
from config_paths import RESULTS_DIR, ESSENTIA_DEAM_HEAD_PB
from embeddings import make_extractor, load_audio_mono
from heads import BiGRUHead, TemporalConvHead, TransformerHead
from metrics import labels_convert


logger = setup_logger()


def pick_head(name: str, in_dim: int):
    n = name.lower().strip()
    if n == "bigru":
        return BiGRUHead(in_dim=in_dim, hidden=128)
    if n == "tcn":
        return TemporalConvHead(in_dim=in_dim, hidden=128)
    if n == "transformer":
        return TransformerHead(in_dim=in_dim, hidden=128)
    raise ValueError(f"Unknown head '{name}'. Choose: bigru | tcn | transformer.")


def main():
    ap = argparse.ArgumentParser(description="Predict dynamic V/A for an audio file (DEAM models).")
    ap.add_argument("audio", help="Audio file path (.mp3/.wav/.flac)")
    ap.add_argument("--model-name", type=str, default="default")
    ap.add_argument("--extractor", default="essentia_musicnn", choices=["essentia_musicnn", "librosa_mfcc_chroma"])
    ap.add_argument("--head", default="bigru", choices=["bigru", "tcn", "transformer"])
    ap.add_argument("--out", default="va_predictions.csv", help="Output CSV with predictions")
    ap.add_argument("--mean", action="store_true", help="Print mean statistics of predictions")
    ap.add_argument("--with-essentia", action="store_true", help="Compare also with Essentia DEAM head")
    ap.add_argument("--out-scale", choices=["norm", "19"], default="19", help="Output scale: 'norm'=[-1,1] or '19'=[1..9] (default: 19)")

    args = ap.parse_args()

    DATASET_NAME = "DEAM"
    results_dir = RESULTS_DIR / DATASET_NAME / args.model_name
    assert results_dir.exists(), f"Run not found: {results_dir}"

    ck = json.loads((results_dir / "best_checkpoints.json").read_text())["fold_checkpoints"]

    y = load_audio_mono(args.audio, sr=16000)
    extractor = make_extractor(args.extractor)
    X = extractor(y, 16000).astype("float32")  # (T, D)
    in_dim = X.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for p in ck:
        sd = torch.load(p, map_location=device)["model"]
        m = pick_head(args.head, in_dim=in_dim).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    with torch.no_grad():
        preds = [m(torch.from_numpy(X).unsqueeze(0).float().to(device)).squeeze(0).cpu().numpy() for m in models]
        P = np.mean(preds, axis=0) if len(preds) > 1 else preds[0]

    P = np.clip(P, -1.0, 1.0).astype("float32")

    out_data = {
        "valence_norm": P[:, 0],
        "arousal_norm": P[:, 1],
        "valence_19": labels_convert(P[:, 0], src="norm", dst="19"),
        "arousal_19": labels_convert(P[:, 1], src="norm", dst="19"),
    }

    if args.with_essentia:
        try:
            head = es.TensorflowPredict2D(
                graphFilename=str(ESSENTIA_DEAM_HEAD_PB),
                output="model/Identity",
            )

            P2 = head(X)  # Essentia: scale [1..9]
            P2_norm = labels_convert(P2, src="19", dst="norm").astype("float32")

            out_data.update({
                "ess_valence_norm": P2_norm[:, 0],
                "ess_arousal_norm": P2_norm[:, 1],
                "ess_valence_19": P2[:, 0],
                "ess_arousal_19": P2[:, 1],
            })

            logger.info("Essentia predictions added.")
        except Exception as e:
            logger.warning(f"Essentia comparison skipped: {e}")

    df = pd.DataFrame(out_data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info(f"Saved predictions to {args.out}")

    if args.mean:
        logger.info(f"[MEAN ours norm] V={df['valence_norm'].mean():.3f} A={df['arousal_norm'].mean():.3f}")
        logger.info(f"[MEAN ours 1..9] V={df['valence_19'].mean():.3f} A={df['arousal_19'].mean():.3f}")
        if args.with_essentia and 'ess_valence_norm' in df.columns:
            logger.info(f"[MEAN ess norm] V={df['ess_valence_norm'].mean():.3f} A={df['ess_arousal_norm'].mean():.3f}")
            logger.info(f"[MEAN ess 1..9] V={df['ess_valence_19'].mean():.3f} A={df['ess_arousal_19'].mean():.3f}")


if __name__ == "__main__":
    main()
