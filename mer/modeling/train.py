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
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from loguru import logger
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import typer

from mer.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, \
    RAW_DATA_DIR
from mer.datasets.common import SongSequenceDataset
from mer.datasets.deam import DEAMDataset
from mer.heads import BiGRUHead
from mer.modeling.utils.loss import make_loss_fn
from mer.modeling.utils.metrics import labels_convert, metrics_dict
from mer.modeling.utils.misc import set_seed, pad_and_mask

app = typer.Typer()


def build_items(
    manifest,
    dataset,
    labels_scale: str = '19'
):
    vmap, amap = dataset.va_maps
    items = []

    for r in tqdm(
        list(manifest.itertuples(index=False)), total=len(manifest), desc='Preparing items'
    ):
        sid = int(r.song_id)

        if not r.annotated:
            logger.error(f'Song {sid} is not annotated, skipping')
            continue

        v, a = vmap[sid], amap[sid]
        L = min(len(v), len(a))
        if L <= 1:
            logger.error(f'Song {sid} has less than 2 frames')
            continue

        Y = np.stack([v[:L], a[:L]], axis=1).astype('float32')
        Y = labels_convert(Y, src=labels_scale, dst='norm').astype('float32')
        X = np.load(r.embeddings_path).astype('float32')

        T = min(len(X), len(Y))
        X, Y = X[:T], Y[:T]

        if len(X) == 0:
            logger.error(f'Song {sid} has no valid frames')
            continue

        items.append((X, Y))
    return items


def evaluate_model(model, dl, device):
    Ys = []
    Ps = []

    model.eval()
    with torch.no_grad():
        for Xb, Yb, Mb in dl:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Mb = Mb.to(device)
            P = model(Xb)
            M = Mb.unsqueeze(-1)
            Ys.append((Yb * M).reshape(-1, 2)[Mb.reshape(-1) > 0].cpu().numpy())
            Ps.append((P * M).reshape(-1, 2)[Mb.reshape(-1) > 0].cpu().numpy())
    Y = np.concatenate(Ys, 0)
    P = np.concatenate(Ps, 0)

    vccc, vp, vr2, vrmse = metrics_dict(Y[:, 0], P[:, 0]).values()
    accc, ap, ar2, armse = metrics_dict(Y[:, 1], P[:, 1]).values()
    return {'Valence_CCC': vccc, 'Valence_Pearson': vp, 'Valence_R2': vr2, 'Valence_RMSE': vrmse,
            'Arousal_CCC': accc, 'Arousal_Pearson': ap, 'Arousal_R2': ar2, 'Arousal_RMSE': armse,
            'CCC_mean': (vccc + accc) / 2.0}


def train_fold(i: int, args, model: nn.Module, train_loader: DataLoader, validate_loader: DataLoader, report_dir):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = make_loss_fn('masked_'+args.loss_type)

    best_path = report_dir / f'fold{i}_best.pt'
    last_path = report_dir / f'fold{i}_last.pt'

    best = -1e9
    patience = args.patience
    history = []

    for epoch in (pbar := tqdm(range(1, args.epochs + 1))):
        pbar.set_postfix_str(f'Epoch {epoch}')

        model.train()
        losses = []

        for Xb, Yb, Mb in train_loader:
            Xb = Xb.to(args.device)
            Yb = Yb.to(args.device)
            Mb = Mb.to(args.device)
            P = model(Xb)
            loss = loss_fn(P, Yb, Mb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses))
        train_m = evaluate_model(model, train_loader, args.device)
        val_m = evaluate_model(model, validate_loader, args.device)

        row = {'epoch': epoch, 'train_loss': train_loss}
        row.update({f'train_{k}': v for k, v in train_m.items()})
        row.update({f'valid_{k}': v for k, v in val_m.items()})
        history.append(row)

        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'epoch': epoch}, last_path)

        score = val_m['CCC_mean']
        if score > best:
            best = score
            patience = args.patience
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'epoch': epoch}, best_path)
        else:
            patience -= 1
            if patience <= 0:
                logger.info(
                    f'[fold {i}] Early stopping at epoch {epoch}')
                break

        logger.info(
            f'[fold {i}][epoch {epoch}] loss={train_loss:.4f} valid_CCC_mean={score:.3f}')

    history = pd.DataFrame(history)
    history.to_csv(report_dir/f'fold{i}_history.csv', index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['train_CCC_mean'], label='train CCC mean')
    plt.plot(history['epoch'], history['valid_CCC_mean'], label='validation CCC mean')
    plt.xlabel('Epoch')
    plt.ylabel('CCC mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir/f'fold{i}_ccc_curve.png')
    plt.close()

    return best_path


@app.command()
def main(
    dataset_name: Annotated[Literal['DEAM',], typer.Option(case_sensitive=False)] = 'DEAM',
    model_path: Path = MODELS_DIR/'model.pkl',
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

    if dataset_name == 'DEAM':
        dataset = DEAMDataset(
            root_dir=RAW_DATA_DIR/dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR/dataset_name/'embeddings'
        )
    else:
        raise NotImplemented(dataset_name)

    report_dir = REPORTS_DIR / f'training_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    report_dir.mkdir()

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
            'validation_ids': song_ids[test].tolist()
        })

    (report_dir / 'splits.json').write_text(json.dumps({
        'dataset_name': dataset_name,
        'model_path': str(model_path),
        'seed': seed,
        'test_ids': test_ids.tolist(),
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
    checkpts = []
    for i, fold in (pbar := tqdm(enumerate(folds, start=1))):
        pbar.set_postfix_str(f'Fold {i}')

        train_manifest = manifest[manifest['song_id'].isin(fold['train_ids'])]
        valid_manifest = manifest[manifest['song_id'].isin(fold['validation_ids'])]

        train_items = build_items(
            manifest=train_manifest, dataset=dataset, labels_scale=labels_scale
        )
        valid_items = build_items(
            manifest=valid_manifest, dataset=dataset, labels_scale=labels_scale
        )
        logger.info(f'Fold {i}: {len(train_items)} train, {len(valid_items)} validation')

        train_dset = SongSequenceDataset(train_items)
        valid_dset = SongSequenceDataset(valid_items)

        # why is shuffle inconsistent?
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True,
                           collate_fn=pad_and_mask)
        valid_loader = DataLoader(valid_dset, batch_size=batch_size,
                           shuffle=False, collate_fn=pad_and_mask)

        model = BiGRUHead(in_dim=train_dset.input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
        best_fold = train_fold(i, model=model, args=args, train_loader=train_loader, validate_loader=valid_loader, report_dir=report_dir)
        checkpts.append(best_fold)

    (report_dir/'best_checkpoints.json').write_text(
        json.dumps(checkpts, indent=2, default=lambda x: str(x))
    )
    logger.success('Modeling training complete.')


if __name__ == "__main__":
    app()
