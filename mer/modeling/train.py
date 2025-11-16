from datetime import datetime
import json
from pathlib import Path
from random import randint
from types import SimpleNamespace
from typing import Annotated, Literal

from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import typer

from mer.config import DEFAULT_DEVICE, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from mer.datasets.common import SongSequenceDataset
from mer.datasets.deam import DEAMDataset
from mer.heads import BiGRUHead
from mer.modeling.utils.loss import make_loss_fn
from mer.modeling.utils.metrics import labels_convert, metrics_dict
from mer.modeling.utils.misc import pad_and_mask, set_seed

app = typer.Typer()


def build_items(manifest, dataset, labels_scale: str = "19"):
    vmap, amap = dataset.va_maps
    items = []

    for r in tqdm(
        manifest.itertuples(index=False), total=len(manifest), desc="Preparing items", leave=False
    ):
        sid = int(r.song_id)

        if not r.annotated:
            logger.error(f"Song {sid} is not annotated, skipping")
            continue

        v, a = vmap[sid], amap[sid]
        L = min(len(v), len(a))
        if L <= 1:
            logger.error(f"Song {sid} has less than 2 frames")
            continue

        Y = np.stack([v[:L], a[:L]], axis=1).astype("float32")
        Y = labels_convert(Y, src=labels_scale, dst="norm").astype("float32")
        X = np.load(r.embeddings_path).astype("float32")

        T = min(len(X), len(Y))
        X, Y = X[:T], Y[:T]

        if len(X) == 0:
            logger.error(f"Song {sid} has no valid frames")
            continue

        items.append((X, Y))
    return items


def _create_history_report(name: str, history: list, report_dir: Path):
    history = pd.DataFrame(history)
    history.to_csv(report_dir / f"{name}_history.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_CCC_mean"], label="train CCC mean")
    plt.plot(history["epoch"], history["valid_CCC_mean"], label="validation CCC mean")
    plt.xlabel("Epoch")
    plt.ylabel("CCC mean")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / f"{name}_ccc_curve.png")
    plt.close()


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
    return {
        "Valence_CCC": vccc,
        "Valence_Pearson": vp,
        "Valence_R2": vr2,
        "Valence_RMSE": vrmse,
        "Arousal_CCC": accc,
        "Arousal_Pearson": ap,
        "Arousal_R2": ar2,
        "Arousal_RMSE": armse,
        "CCC_mean": (vccc + accc) / 2.0,
    }


def train_model(
    name: str,
    args: SimpleNamespace,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    report_dir: Path,
    writer: SummaryWriter,
):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     opt, mode='max', factor=0.5, patience=5
    # )
    loss_fn = make_loss_fn("masked_" + args.loss_type)

    best_score = -1e9
    best_m = {}
    patience_counter = 0
    history = []

    for epoch in (pbar := tqdm(range(1, args.epochs + 1), leave=False)):
        pbar.set_postfix_str(f"Epoch {epoch}")

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
        test_m = evaluate_model(model, test_loader, args.device)

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"valid_{k}": v for k, v in test_m.items()})
        history.append(row)

        # LOGING TO TENSORBOARD
        writer.add_scalar(f"{name}/Loss/train", train_loss, epoch)
        writer.add_scalars(
            f"{name}/CCC_mean", {"train": train_m["CCC_mean"], "valid": test_m["CCC_mean"]}, epoch
        )

        writer.add_scalars(
            f"{name}/Valence/metrics",
            {
                "CCC": test_m["Valence_CCC"],
                "Pearson": test_m["Valence_Pearson"],
                "R2": test_m["Valence_R2"],
                "RMSE": test_m["Valence_RMSE"],
            },
            epoch,
        )

        writer.add_scalars(
            f"{name}/Arousal/metrics",
            {
                "CCC": test_m["Arousal_CCC"],
                "Pearson": test_m["Arousal_Pearson"],
                "R2": test_m["Arousal_R2"],
                "RMSE": test_m["Arousal_RMSE"],
            },
            epoch,
        )

        score = test_m["CCC_mean"]
        # scheduler.step(score)
        # logger.debug(f'{scheduler.get_last_lr()=}')

        if score > best_score:
            best_score = score
            best_m = test_m
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"[{name}] Early stopping at epoch {epoch}")
                break

        logger.info(f"[{name}][epoch {epoch}] loss={train_loss:.4f} valid_CCC_mean={score:.3f}")

    _create_history_report(name, history, report_dir)
    return model, best_m


@app.command()
def main(
    dataset_name: Annotated[Literal["DEAM",], typer.Option(case_sensitive=False)] = "DEAM",
    # model_path: Path = MODELS_DIR / "model.pth",
    head: Annotated[Literal["BiGRU",], typer.Option(case_sensitive=False)] = "BiGRU",
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 6,
    patience: int = 15,
    kfolds: int = 5,
    seed: int = randint(1, 1000000),
    test_size: float = 0.1,
    loss_type: Annotated[
        Literal["ccc", "mse", "hybrid"], typer.Option(case_sensitive=False)
    ] = "ccc",
    hidden_dim: int = 128,
    dropout: float = 0.2,
    labels_scale: Annotated[
        Literal["19", "norm"],
        typer.Option(case_sensitive=False, help="Scale of dynamic labels in DEAM source files"),
    ] = "norm",
    device: Annotated[Literal["cuda", "cpu"], typer.Option(case_sensitive=False)] = DEFAULT_DEVICE,
):
    args = SimpleNamespace(**locals())

    if dataset_name == "DEAM":
        dataset = DEAMDataset(
            root_dir=RAW_DATA_DIR / dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR / dataset_name / "embeddings",
        )
    else:
        raise NotImplemented(dataset_name)

    report_dir = REPORTS_DIR / f'training_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    report_dir.mkdir()
    writer = SummaryWriter(log_dir=str(report_dir / "tensorboard"))

    embeddings_dir = PROCESSED_DATA_DIR / dataset_name / "embeddings"
    assert embeddings_dir.is_dir(), "Embeddings dir not found"

    manifest_path = PROCESSED_DATA_DIR / dataset_name / "manifest.csv"
    assert manifest_path.is_file(), "Manifest file not found"
    manifest = pd.read_csv(manifest_path)

    set_seed(args.seed)

    n_samples = len(manifest)
    indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True
    )
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in kf.split(train_indices):
        folds.append({"train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist()})

    (report_dir / "splits.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                # 'model_path': str(model_path),
                "seed": seed,
                "folds": folds,
                "head": head,
                "train_params": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "patience": patience,
                    "loss_type": loss_type,
                    "hidden": hidden_dim,
                    "dropout": dropout,
                },
            },
            indent=2,
        )
    )

    logger.info(f"Training started. Report dir: {report_dir}")

    items = build_items(manifest=manifest, dataset=dataset, labels_scale=args.labels_scale)
    dset = SongSequenceDataset(items)

    # K-fold validation for performance estimation only
    fold_scores = []
    for i, fold in enumerate(folds, start=1):
        logger.info(f"Training fold {i}...")

        train_subset = Subset(dset, indices=fold["train_indices"])
        valid_subset = Subset(dset, indices=fold["validation_indices"])
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, collate_fn=pad_and_mask
        )
        valid_loader = DataLoader(
            valid_subset, batch_size=args.batch_size, collate_fn=pad_and_mask
        )
        logger.info(f"Fold {i} size: {len(train_subset)} train, {len(valid_subset)} validation")

        model = BiGRUHead(in_dim=dset.input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)

        # Add model graph to TensorBoard
        if i == 1:
            Xb, Yb = valid_subset[0]
            Xb = torch.tensor(Xb, dtype=torch.float32).unsqueeze(0).to(device)
            writer.add_graph(model, Xb)

        _, score = train_model(
            name=f"fold_{i}",
            model=model,
            args=args,
            train_loader=train_loader,
            test_loader=valid_loader,
            report_dir=report_dir,
            writer=writer,
        )
        fold_scores.append(score["CCC_mean"])

    # Report k-fold validation results
    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    logger.info(f"K-fold validation CCC: {avg_score:.4f} ± {std_score:.4f}")

    logger.info("Training final model on full training set...")

    train_subset = Subset(dset, indices=train_indices)
    test_subset = Subset(dset, indices=test_indices)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, collate_fn=pad_and_mask)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, collate_fn=pad_and_mask)
    logger.info(f"Final size: {len(train_subset)} train, {len(test_subset)} test")

    model = BiGRUHead(in_dim=dset.input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)

    model, score = train_model(
        name="final",
        model=model,
        args=args,
        train_loader=train_loader,
        test_loader=test_loader,
        report_dir=report_dir,
        writer=writer,
    )
    model_path = report_dir / "model.pth"
    torch.save(model, model_path)

    # Write final scores
    table = "| Metric | Value |\n|:--|--:|\n"
    for key, value in score.items():
        table += f"| {key} | {value:.4f} |\n"

    writer.add_text("Test results", table)
    writer.close()

    (report_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "training_size": len(train_subset),
                "test_size": len(test_subset),
                "test_score": score["CCC_mean"],
                "kfold_validation_score_mean": avg_score,
                "kfold_validation_score_std": std_score,
                "dataset": dataset_name,
                "hyperparameters": {
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "lr": lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "loss_type": loss_type,
                },
            },
            indent=2,
        )
    )

    logger.success("Model training complete.")


if __name__ == "__main__":
    app()
