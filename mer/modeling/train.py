import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal
from random import randint

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import typer

from mer.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from mer.modeling.utils.loss import make_loss_fn
from mer.modeling.utils.metrics import labels_convert
from mer.modeling.utils.misc import set_seed


app = typer.Typer()


class SongSequenceDataset(Dataset):
    def __init__(self, items: list[tuple[np.ndarray, np.ndarray]]):
        self.items = items
        self.input_dim = items[0][0].shape[1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_items(
    df,
    dataset: DEAMDataset,
    extractor,
    split_label: str = "",
    cached_only: bool = False,
    labels_scale: str = "19"
):
    vmap, amap = dataset.load_dynamic_va_maps()

    items = []
    failures = []
    cached = computed = failed = skipped = 0

    it = tqdm(
        list(df.itertuples(index=False)),
        total=len(df),
        desc=f"Preparing {split_label or 'items'}",
        smoothing=0.1,
    )

    for r in it:
        sid = int(r.song_id)
        apath = Path(r.audio_path)
        cfile = dataset.cache_file(sid, "librosa")



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


def train_fold(args, model: nn.Module, train_loader: DataLoader, validate_loader: DataLoader, report_dir):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = make_loss_fn('masked_'+args.loss_type)

    best_path = report_dir / f'fold{fold_idx}_best.pt'
    last_path = report_dir / f'fold{fold_idx}_last.pt'

    best = -1e9
    patience = args.patience
    history = []

    for epoch in (pbar := tqdm(range(1, args.epochs + 1))):
        pbar.set_postfix_str(f'Epoch {epoch}')

        model.train()
        losses = []
        for Xb, Yb, Mb in train_loader:
            Xb = Xb.to(args.device)  # is it necessary when the model has already '.to(device)'?
            Yb = Yb.to(args.device)
            Mb = Mb.to(args.device)
            P = model(Xb)
            loss = loss_fn(P, Yb, Mb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        train_m = evaluate_model(model, train_loader, device)
        val_m = evaluate_model(model, validate_loader, device)

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}": v for k, v in val_m.items()})
        history.append(row)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch}, last_path)

        score = val_m["CCC_mean"]
        if score > best:
            best = score
            patience = args.patience
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch}, best_path)
        else:
            patience -= 1
            if patience <= 0:
                logger.info(
                    f"[fold {fold_idx}] Early stopping at epoch {epoch}")
                break

        logger.info(
            f"[fold {fold_idx}][epoch {epoch}] loss={train_loss:.4f} val_CCC_mean={score:.3f}")

    df = pd.DataFrame(history)
    df.to_csv(self.logs_dir / f"fold{fold_idx}_history.csv", index=False)
    plt.figure(figsize=(10, 5))
    if "train_CCC_mean" in df and "val_CCC_mean" in df:
        plt.plot(df["epoch"], df["train_CCC_mean"], label="train CCC mean")
        plt.plot(df["epoch"], df["val_CCC_mean"], label="val CCC mean")
    plt.xlabel("Epoch")
    plt.ylabel("CCC mean")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(self.plots_dir / f"fold{fold_idx}_ccc_curve.png")
    plt.close()

    return best_path


@app.command()
def main(
    dataset_name: Annotated[Literal['DEAM',], typer.Option(case_sensitive=False)] = 'DEAM',
    model_path: Path = MODELS_DIR / 'model.pkl',
    head: Annotated[Literal['BiGRU',], typer.Option(case_sensitive=False)] = 'BiGRU',
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 6,
    patience: int = 5,
    kfolds: int = 5,
    seed: int = randint(1, 1000000),
    test_size: float = 0.1,
    loss_type: Annotated[Literal['ccc', 'mse', 'hybrid'], typer.Option(case_sensitive=False)] = 'ccc',
    hidden_dim: int = 128,
    dropout: float = 0.2,
    labels_scale: Annotated[Literal['19', 'norm'], typer.Option(case_sensitive=False, help='Scale of dynamic labels in DEAM source files')] = 'norm',
    device: Annotated[Literal['cuda', 'cpu'], typer.Option(case_sensitive=False)] = ('cuda' if torch.cuda.is_available() else 'cpu'),
):
    args = SimpleNamespace(**locals())
    report_dir = REPORTS_DIR / f'training_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    embeddings_dir = PROCESSED_DATA_DIR / dataset_name / 'embeddings'
    assert embeddings_dir.is_dir(), 'Embeddings dir not found'

    manifest_path = PROCESSED_DATA_DIR / dataset_name / 'manifest.csv'
    assert manifest_path.is_file(), 'Manifest file not found'
    manifest = pd.read_csv(manifest_path)

    set_seed(args.seed)

    song_ids = manifest['song_id'].values
    train_ids, test_ids = train_test_split(
        song_ids, test_size=test_size, random_state=seed, shuffle=True
    )
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

    folds = []
    for train, test in kf.split(train_ids):
        folds.append({
            'train_ids': song_ids[train].tolist(),
            'val_ids': song_ids[test].tolist()
        })

    (report_dir / 'splits.json').write_text(json.dumps({
        'dataset_name': dataset_name,
        'model_path': model_path,
        'seed': seed,
        'test_ids': test_ids,
        'folds': folds,
        'head': head,
        'train_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'patience': patience,
            'loss_type': loss_type,
            'hidden': hidden_dim,
            'dropout': dropout
        }
    }, indent=2))

    logger.info(f'Training started. Report dir: {report_dir}')
    for i, fold in (pbar := tqdm(enumerate(folds, start=1))):
        pbar.set_postfix_str(f'Fold {i}')

        df_tr = manifest[manifest['song_id'].isin(fold['train_ids'])]
        df_va = manifest[manifest['song_id'].isin(fold['val_ids'])]
        train_items, train_fail = build_items(
            df_tr, dataset, extractor,
            split_label='train',
            cached_only=args.cached_only,
            labels_scale=args.labels_scale
        )
        val_items, val_fail = build_items(
            df_va, dataset, extractor,
            split_label='val',
            cached_only=args.cached_only,
            labels_scale=args.labels_scale
        )

        ds_tr = SongSequenceDataset(train_items)
        ds_va = SongSequenceDataset(val_items)
        dl_tr = DataLoader(ds_tr, batch_size=self.cfg.batch_size, shuffle=True,
                           collate_fn=pad_and_mask)
        dl_va = DataLoader(ds_va, batch_size=self.cfg.batch_size,
                           shuffle=False, collate_fn=pad_and_mask)

        model = model_builder(ds_tr.input_dim).to(device)
        train_fold(model=model, args=args, train_loader=dl_tr, validate_loader=dl_va, report_dir=report_dir)

    logger.success('Modeling training complete.')


if __name__ == "__main__":
    app()
