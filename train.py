from __future__ import annotations

import argparse
import json
import numpy as np

from pathlib import Path
from tqdm import tqdm

from logging_utils import setup_logger
from config_paths import RESULTS_DIR
from datasets import DEAMDataset
from embeddings import make_extractor, load_audio_mono
from heads import BiGRUHead, TemporalConvHead, TransformerHead
from metrics import labels_convert
from trainer import Trainer, TrainConfig


logger = setup_logger()


def build_items(
    df, 
    dataset: DEAMDataset, 
    extractor,
    extractor_name: str,
    split_label: str = "", 
    cached_only: bool = False, 
    labels_scale: str = "19"
):
    vmap, amap = dataset.load_dynamic_va_maps()
    prefer_librosa = getattr(extractor, "name", "") == "librosa_mfcc_chroma"

    items = []
    failures = []
    cached = computed = failed = skipped = 0

    it = tqdm(
        list(df.itertuples(index=False)),
        total=len(df),
        desc=f"Preparing {split_label or 'items'} [{extractor_name}]",
        smoothing=0.1,
    )

    for r in it:
        sid = int(r.song_id)
        apath = Path(r.audio_path)
        cfile = dataset.cache_file(sid, extractor_name)

        try:
            if cfile.exists():
                X = np.load(cfile)
                cached += 1
            else:
                if cached_only:
                    skipped += 1
                    it.set_postfix(cached=cached, computed=computed, failed=failed, skipped=skipped)
                    continue
                y = load_audio_mono(apath, sr=16000, prefer_librosa=prefer_librosa)
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
        Y = np.stack([v[:L], a[:L]], axis=1).astype("float32")
        Y = labels_convert(Y, src=labels_scale, dst='norm').astype("float32")
        X = X.astype("float32")

        T = min(len(X), len(Y))
        X, Y = X[:T], Y[:T]

        if len(X) == 0:
            skipped += 1
            it.set_postfix(cached=cached, computed=computed, failed=failed, skipped=skipped)
            continue

        items.append((X, Y))
        it.set_postfix(cached=cached, computed=computed, failed=failed, skipped=skipped)

    logger.info(
        f"[{split_label or 'items'}] cached={cached} computed={computed} "
        f"failed={failed} skipped={skipped} kept={len(items)}"
    )
    return items, failures


def pick_head(name: str, in_dim: int, hidden: int, dropout: float):
    n = name.lower().strip()
    if n == "bigru":       
        return BiGRUHead(in_dim=in_dim, hidden=hidden, dropout=dropout)
    if n == "tcn":
        return TemporalConvHead(in_dim=in_dim, hidden=hidden, dropout=dropout)
    if n == "transformer":
        return TransformerHead(in_dim=in_dim, hidden=hidden, dropout=dropout)
    raise ValueError(f"Unknown head '{name}'. Use: bigru | tcn | transformer.")


def main():
    ap = argparse.ArgumentParser(description="Train dynamic VA on DEAM (run-named outputs).")
    ap.add_argument("--model-name", type=str, default="default", help="Run/model name for results directory.")
    ap.add_argument("--extractor", default="essentia_musicnn", choices=["essentia_musicnn", "librosa_mfcc_chroma"])
    ap.add_argument("--head", default="bigru", choices=["bigru", "tcn", "transformer"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--kf", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--test_size", type=float, default=0.10)
    ap.add_argument("--loss", type=str, default="ccc", choices=["mse", "ccc", "hybrid"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--labels-scale", choices=["19", "norm"], default="norm", help="Scale of dynamic labels in DEAM source files (default: norm).")
    args = ap.parse_args()

    DATASET_NAME = "DEAM"
    results_dir = RESULTS_DIR / DATASET_NAME / args.model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = DEAMDataset()
    manifest = dataset.build_manifest()
    logger.info(f"Manifest size (with annotations): {len(manifest)}")

    from sklearn.model_selection import train_test_split, KFold
    ids = sorted(manifest["song_id"].unique().tolist())
    train_ids, test_ids = train_test_split(ids, test_size=args.test_size, random_state=args.seed, shuffle=True)
    kf = KFold(n_splits=args.kf, shuffle=True, random_state=args.seed)
    folds = []
    tidx = np.array(train_ids)
    for tr, va in kf.split(tidx):
        folds.append({"train_ids": tidx[tr].tolist(), "val_ids": tidx[va].tolist()})

    (results_dir / "manifest.csv").write_text(manifest.to_csv(index=False))
    (results_dir / "splits.json").write_text(json.dumps({
        "dataset": DATASET_NAME,
        "model_name": args.model_name,
        "seed": args.seed,
        "test_ids": test_ids,
        "folds": folds,
        "extractor": args.extractor,
        "head": args.head,
        "train_params": {
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "patience": args.patience,
            "loss": args.loss,
            "hidden": args.hidden,
            "dropout": args.dropout
        }
    }, indent=2))

    extractor = make_extractor(args.extractor)
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        patience=args.patience,
        seed=args.seed,
        loss_type=args.loss,
    )
    trainer = Trainer(results_dir, cfg)

    ckpts = []
    for i, fold in enumerate(folds, start=1):
        df_tr = manifest[manifest["song_id"].isin(fold["train_ids"])]
        df_va = manifest[manifest["song_id"].isin(fold["val_ids"])]
        train_items, train_fail = build_items(
            df_tr, dataset, extractor, args.extractor,
            split_label="train",
            cached_only=args.cached_only,
            labels_scale=args.labels_scale
        )
        val_items, val_fail = build_items(
            df_va, dataset, extractor, args.extractor,
            split_label="val",
            cached_only=args.cached_only,
            labels_scale=args.labels_scale
        )
        # in_dim = train_items[0][0].shape[1]

        def build_model(_in):  # _in == in_dim
            return pick_head(args.head, in_dim=_in, hidden=args.hidden, dropout=args.dropout)

        best = trainer.train_fold(i, train_items, val_items, build_model)
        ckpts.append(best)

    (results_dir / "best_checkpoints.json").write_text(json.dumps({
        "seed": args.seed,
        "fold_checkpoints": ckpts
    }, indent=2))

    logger.info(f"Training completed. Run saved to: {results_dir}")


if __name__ == "__main__":
    main()
