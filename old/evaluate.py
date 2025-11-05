from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from tqdm import tqdm

from logging_utils import setup_logger
from config_paths import RESULTS_DIR
from datasets import DEAMDataset
from embeddings import LibrosaMFCCChroma, load_audio_mono
from heads import BiGRUHead
from metrics import metrics_dict, labels_convert


logger = setup_logger()


def scatter(gt, pr, title, out_png):
    plt.figure(figsize=(5, 5))
    plt.scatter(gt, pr, s=8, alpha=0.4)
    lo = float(min(gt.min(), pr.min()))
    hi = float(max(gt.max(), pr.max()))
    plt.plot([lo, hi], [lo, hi], lw=1)
    plt.title(title)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def pick_head(in_dim: int, hidden: int):
    return BiGRUHead(in_dim=in_dim, hidden=hidden)


def assert_scale(y: np.ndarray, declared: str):
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if declared == "norm":
        if not (-1.25 <= ymin <= 1.25 and -1.25 <= ymax <= 1.25):
            raise ValueError(
                f"Labels expected in 'norm' [-1,1], but got min={ymin:.3f}, max={ymax:.3f}. Try with --labels-scale 19."
            )
    elif declared == "19":
        if not (0.5 <= ymin <= 9.5 and 0.5 <= ymax <= 9.5):
            raise ValueError(
                f"Labels expected in '19' [1,9], but got min={ymin:.3f}, max={ymax:.3f}. Try with --labels-scale norm."
            )
    else:
        raise ValueError(f"Unsupported labels-scale '{declared}'.")


def main():
    ap = argparse.ArgumentParser(description="Evaluate a named run on DEAM with detailed caching logs.")
    ap.add_argument("--model-name", type=str, default="default")
    ap.add_argument("--head", default="bigru", choices=["bigru"])
    ap.add_argument("--labels-scale", choices=["19", "norm"], default="norm", help="Scale of dynamic labels in DEAM source files.")
    ap.add_argument("--plots-scale", choices=["19", "norm"], default="19", help="Scale used for scatter plots (default: 19).")
    args = ap.parse_args()

    DATASET_NAME = "DEAM"
    results_dir = RESULTS_DIR / DATASET_NAME / args.model_name
    assert results_dir.exists(), f"Run not found: {results_dir}"

    dset = DEAMDataset()
    manifest = pd.read_csv(results_dir / "manifest.csv")
    splits = json.loads((results_dir / "splits.json").read_text())
    test_ids = splits["test_ids"]

    extractor = LibrosaMFCCChroma()

    vmap, amap = dset.load_dynamic_va_maps()
    test_df = manifest[manifest["song_id"].isin(test_ids)]
    items, cached, computed, failed, skipped = [], 0, 0, 0, 0
    failures = []

    it = tqdm(
        list(test_df.itertuples(index=False)),
        total=len(test_df),
        desc=f"Preparing TEST",
        smoothing=0.1,
    )

    for r in it:
        sid = int(r.song_id)
        apath = Path(r.audio_path)
        cfile = dset.cache_file(sid, "librosa")

        try:
            if cfile.exists():
                X = np.load(cfile)
                cached += 1
            else:
                y = load_audio_mono(apath, sr=16000)
                X = extractor(y, 16000)
                cfile.parent.mkdir(parents=True, exist_ok=True)
                np.save(cfile, X.astype("float32"))
                computed += 1
        except Exception as e:
            failed += 1
            failures.append((sid, str(apath), str(e)))
            it.set_postfix(cached=cached, computed=computed, failed=failed, skipped=skipped)
            continue

        if sid not in vmap or sid not in amap:
            skipped += 1
            it.set_postfix(cached=cached, computed=computed, failed=failed, skipped=skipped)
            continue
        v, a = vmap[sid], amap[sid]
        L = min(len(v), len(a))
        if L <= 1:
            skipped += 1
            it.set_postfix(cached=cached, computed=computed, failed=failed, skipped=skipped)
            continue

        Y_raw = np.stack([v[:L], a[:L]], axis=1).astype("float32")

        assert_scale(Y_raw, args.labels_scale)
        Y = labels_convert(Y_raw, src=args.labels_scale, dst="norm").astype("float32")

        T = min(len(X), len(Y))
        X, Y = X[:T], Y[:T]

        if len(X) == 0:
            skipped += 1
            it.set_postfix(cached=cached, computed=computed, failed=failed, skipped=skipped)
            continue

        items.append((sid, X, Y))
        it.set_postfix(cached=cached, computed=computed, failed=failed, skipped=skipped)

    logger.info(f"[TEST] cached={cached} computed={computed} failed={failed} skipped={skipped} kept={len(items)}")
    if failures:
        fail_csv = results_dir / f"eval_cache_failures_{args.extractor}.csv"
        pd.DataFrame(failures, columns=["song_id", "audio_path", "reason"]).to_csv(fail_csv, index=False)
        logger.warning(f"Wrote failures report: {fail_csv}")

    ck = json.loads((results_dir / "best_checkpoints.json").read_text())["fold_checkpoints"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = items[0][1].shape[1]
    models = []
    for p in ck:
        sd = torch.load(p, map_location=device)["model"]
        m = pick_head(in_dim=in_dim, hidden=128).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    Y_all, P_all = [], []
    for sid, X, Y in items:
        with torch.no_grad():
            preds = [m(torch.from_numpy(X).unsqueeze(0).float().to(device)).squeeze(0).cpu().numpy() for m in models]
            P = np.mean(preds, 0) if len(preds) > 1 else preds[0]
        Y_all.append(Y)
        P_all.append(P)
    Y = np.concatenate(Y_all, 0)
    P = np.concatenate(P_all, 0)

    mV = metrics_dict(Y[:, 0], P[:, 0])
    mA = metrics_dict(Y[:, 1], P[:, 1])
    rows = [{
        "system": "ours",
        "CCC_valence": mV["CCC"], "Pearson_valence": mV["Pearson"], "R2_valence": mV["R2"], "RMSE_valence": mV["RMSE"],
        "CCC_arousal": mA["CCC"], "Pearson_arousal": mA["Pearson"], "R2_arousal": mA["R2"], "RMSE_arousal": mA["RMSE"],
        "CCC_mean": (mV["CCC"] + mA["CCC"]) / 2.0
    }]

    out_plots = results_dir / "plots"
    out_plots.mkdir(exist_ok=True, parents=True)
    if args.plots_scale == "19":
        Y_plot = labels_convert(Y, src="norm", dst="19")
        P_plot = labels_convert(P, src="norm", dst="19")
    else:
        Y_plot, P_plot = Y, P

    scatter(Y_plot[:, 0], P_plot[:, 0], "Valence — Ours (test)", out_plots / "scatter_valence_ours.png")
    scatter(Y_plot[:, 1], P_plot[:, 1], "Arousal — Ours (test)", out_plots / "scatter_arousal_ours.png")

    pd.DataFrame(rows).to_csv(results_dir / "test_metrics_compare.csv", index=False)
    logger.info(f"Saved evaluation results and plots to: {results_dir}")


if __name__ == "__main__":
    main()
