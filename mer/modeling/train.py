import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal

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

app = typer.Typer()
is_cuda_available = torch.cuda.is_available()


class SongSequenceDataset(Dataset):
    def __init__(self, items: list[tuple[np.ndarray, np.ndarray]]):
        self.items = items
        self.input_dim = items[0][0].shape[1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def train_fold(args, model: nn.Module, train_loader: DataLoader, validate_loader: DataLoader):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = make_loss_fn(args.loss_type)
    best = -1e9
    patience = args.patience
    best_path = self.ckpt_dir / f"fold{fold_idx}_best.pt"
    last_path = self.ckpt_dir / f"fold{fold_idx}_last.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
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
        train_m = evaluate_model(model, dl_tr, device)
        val_m = evaluate_model(model, dl_va, device)

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
    seed: int = 2024,
    test_size: float = 0.1,
    loss_type: Annotated[Literal['ccc', 'mse', 'hybrid'], typer.Option(case_sensitive=False)] = 'ccc',
    hidden_dim: int = 128,
    dropout: float = 0.2,
    labels_scale: Annotated[Literal['19', 'norm'], typer.Option(case_sensitive=False, help='Scale of dynamic labels in DEAM source files')] = 'norm',
    device: Annotated[Literal['cuda', 'cpu'], typer.Option(case_sensitive=False)] = ('cuda' if is_cuda_available else 'cpu'),
):
    args = SimpleNamespace(**locals())
    report_dir = REPORTS_DIR / f'training_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    embeddings_dir = PROCESSED_DATA_DIR / dataset_name / 'embeddings'
    assert embeddings_dir.is_dir(), 'Embeddings dir not found'

    manifest_path = PROCESSED_DATA_DIR / dataset_name / 'manifest.csv'
    assert manifest_path.is_file(), 'Manifest file not found'
    manifest = pd.read_csv(manifest_path)

    logger.info(f'Manifest shape: {manifest.shape}')
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
        'dataset': dataset_name,
        'model_name': model_path.stem,
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

    for i, fold in enumerate(folds, start=1):
        train_fold(args)

    logger.success('Modeling training complete.')


if __name__ == "__main__":
    app()
