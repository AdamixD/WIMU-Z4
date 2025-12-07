"""
Unified training script for all datasets (DEAM, PMEmo, MERGE) with support for:
- VA regression mode: Continuous valence-arousal prediction
- Russell 4Q classification mode: 4-quadrant emotion classification
"""

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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import typer

from mer.config import DEFAULT_DEVICE, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from mer.datasets.common import SongClassificationDataset, SongSequenceDataset
from mer.datasets.deam import DEAMDataset
from mer.datasets.merge import MERGEDataset
from mer.datasets.pmemo import PMEmoDataset
from mer.heads import BiGRUClassificationHead, BiGRUHead
from mer.modeling.utils.data_loaders import (
    build_items_classification,
    build_items_merge_classification,
    build_items_merge_regression,
    build_items_regression,
)
from mer.modeling.utils.loss import make_loss_fn
from mer.modeling.utils.metrics import (
    classification_metrics,
    labels_convert,
    metrics_dict,
)
from mer.modeling.utils.misc import pad_and_mask, pad_and_mask_classification, set_seed
from mer.modeling.utils.report import save_splits, save_training_summary
from mer.modeling.utils.train_utils import (
    prepare_kfold,
    prepare_kfold_dataset,
    prepare_datasets,
    prepare_datasets_merge,
    get_mode_components
)

app = typer.Typer()


def _create_history_report_regression(name: str, history: list, report_dir: Path):
    """Create history report for regression mode"""
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


def _create_history_report_classification(name: str, history: list, report_dir: Path):
    """Create history report for classification mode"""
    history = pd.DataFrame(history)
    history.to_csv(report_dir / f"{name}_history.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_Accuracy"], label="train accuracy")
    plt.plot(history["epoch"], history["valid_Accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_dir / f"{name}_accuracy_curve.png")
    plt.close()


def _create_scatter_plot(
    Y_true: np.ndarray, Y_pred: np.ndarray, writer: SummaryWriter, report_dir: Path
):
    """Create scatter plots for VA regression"""
    for i, dim in enumerate(["valence", "arousal"]):
        p = plt.figure(figsize=(5, 5))
        plt.scatter(Y_true[:, i], Y_pred[:, i], s=8, alpha=0.4)
        lo = float(min(Y_true[:, i].min(), Y_pred[:, i].min()))
        hi = float(max(Y_true[:, i].max(), Y_pred[:, i].max()))
        plt.plot([lo, hi], [lo, hi], lw=1)
        plt.title(f"{dim}: ground truth vs prediction")
        plt.xlabel("Ground Truth")
        plt.ylabel("Prediction")
        plt.tight_layout()
        plt.savefig(report_dir / f"scatter_{dim}.png")

        writer.add_figure(f"Scatter/{dim}", p)

        plt.close()


def _create_confusion_matrix(
    Y_true: np.ndarray, Y_pred: np.ndarray, writer: SummaryWriter, report_dir: Path
):
    """Create confusion matrix for classification"""
    cm = confusion_matrix(Y_true, Y_pred, labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    quadrant_labels = ["Q1\n(Happy)", "Q2\n(Angry)", "Q3\n(Sad)", "Q4\n(Calm)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=quadrant_labels,
        yticklabels=quadrant_labels,
        title="Confusion Matrix - Russell 4Q",
        ylabel="True Quadrant",
        xlabel="Predicted Quadrant",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png")

    writer.add_figure("Confusion Matrix", fig)
    plt.close()


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def evaluate_model_regression(model, dl, device, writer=None, report_dir=None, scatter=False):
    """Evaluate model in VA regression mode"""
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

    if scatter and writer and report_dir:
        _create_scatter_plot(Y, P, writer, report_dir)

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


def evaluate_model_classification(
    model, dl, device, writer=None, report_dir=None, confusion=False
):
    """Evaluate model in Russell 4Q classification mode"""
    Ys = []
    Ps = []

    model.eval()
    with torch.no_grad():
        for Xb, Yb, Mb in dl:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Mb = Mb.to(device)

            logits = model(Xb)
            pred = torch.argmax(logits, dim=-1)

            valid = Mb > 0
            Ys.append(Yb[valid].cpu().numpy())
            Ps.append(pred[valid].cpu().numpy())

    Y = np.concatenate(Ys, 0)
    P = np.concatenate(Ps, 0)

    if confusion and writer and report_dir:
        _create_confusion_matrix(Y, P, writer, report_dir)

    return classification_metrics(Y, P)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def train_model_regression(
    name: str,
    args: SimpleNamespace,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    report_dir: Path,
    writer: SummaryWriter,
    scatter: bool = False,
):
    """Train model in VA regression mode"""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = make_loss_fn("masked_" + args.loss_type)

    best_score = -1e9
    best_m = {}
    patience_counter = 0
    history = []

    for epoch in (pbar := tqdm(range(1, args.epochs + 1), leave=False)) :
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
        train_m = evaluate_model_regression(model, train_loader, args.device)
        test_m = evaluate_model_regression(
            model, test_loader, args.device, writer, report_dir, scatter
        )

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"valid_{k}": v for k, v in test_m.items()})
        history.append(row)

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

    _create_history_report_regression(name, history, report_dir)
    return model, best_m


def train_model_classification(
    name: str,
    args: SimpleNamespace,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    report_dir: Path,
    writer: SummaryWriter,
    confusion: bool = False,
):
    """Train model in Russell 4Q classification mode"""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = make_loss_fn("masked_cross_entropy")

    best_score = -1e9
    best_m = {}
    patience_counter = 0
    history = []

    for epoch in (pbar := tqdm(range(1, args.epochs + 1), leave=False)) :
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
        train_m = evaluate_model_classification(model, train_loader, args.device)
        test_m = evaluate_model_classification(
            model, test_loader, args.device, writer, report_dir, confusion
        )

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"valid_{k}": v for k, v in test_m.items()})
        history.append(row)

        writer.add_scalar(f"{name}/Loss/train", train_loss, epoch)
        writer.add_scalars(
            f"{name}/Metrics", {"Accuracy": test_m["Accuracy"], "F1": test_m["F1"],}, epoch,
        )

        score = test_m["Accuracy"]

        if score > best_score:
            best_score = score
            best_m = test_m
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"[{name}] Early stopping at epoch {epoch}")
                break

        logger.info(f"[{name}][epoch {epoch}] loss={train_loss:.4f} valid_accuracy={score:.3f}")

    _create_history_report_classification(name, history, report_dir)
    return model, best_m


# ============================================================================
# MAIN TRAINING COMMAND
# ============================================================================


@app.command()
def main(
    dataset_name: Annotated[
        Literal["DEAM", "PMEmo", "MERGE"], typer.Option(case_sensitive=False)
    ] = "DEAM",
    prediction_mode: Annotated[
        Literal["VA", "Russell4Q"],
        typer.Option(case_sensitive=False, help="VA for regression, Russell4Q for classification"),
    ] = "VA",
    head: Annotated[Literal["BiGRU"], typer.Option(case_sensitive=False)] = "BiGRU",
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
    merge_split: Annotated[
        Literal["70_15_15", "40_30_30"],
        typer.Option(case_sensitive=False, help="MERGE dataset split ratio"),
    ] = "70_15_15",
    device: Annotated[Literal["cuda", "cpu"], typer.Option(case_sensitive=False)] = DEFAULT_DEVICE,
    augment: Annotated[
    Literal[None, "shift", "gain", "reverb", "lowpass", "highpass", "bandpass", "pitch_shift"], typer.Option(case_sensitive=False)
    ] = None,
    augment_size: float = 0.3,
):
    """
    Train model on DEAM, PMEmo, or MERGE dataset.
    
    For MERGE dataset:
    - prediction_mode='VA': Train regression model for continuous valence-arousal
    - prediction_mode='Russell4Q': Train classification model for 4-quadrant emotions
    
    For DEAM/PMEmo datasets:
    - Only VA mode is supported (regression)
    """
    args = SimpleNamespace(**locals())

    # Validate prediction mode
    if prediction_mode not in ["VA", "Russell4Q"]:
        raise ValueError("prediction_mode must be 'VA' or 'Russell4Q'")

    # Info about automatic label generation for Russell4Q
    if dataset_name in ["DEAM", "PMEmo"] and prediction_mode == "Russell4Q":
        logger.info(
            f"Russell4Q mode on {dataset_name}: Labels will be automatically generated from VA values"
        )

    # Initialize dataset
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
    elif dataset_name == "MERGE":
        dataset = MERGEDataset(
            root_dir=RAW_DATA_DIR / dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR / dataset_name / "embeddings",
            mode=prediction_mode,
        )
    else:
        raise NotImplementedError(dataset_name)

    suffix = f"_{augment}" if augment is not None else ""
    report_dir = (
        REPORTS_DIR
        / f'training_{dataset_name}_{prediction_mode}{suffix}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    report_dir.mkdir(parents=True)
    writer = SummaryWriter(log_dir=str(report_dir / "tensorboard"))

    embeddings_dir = PROCESSED_DATA_DIR / dataset_name / "embeddings"
    assert embeddings_dir.is_dir(), f"Embeddings dir not found: {embeddings_dir}"
    aug_manifest = None
    if augment:
        aug_embeddings_dir = PROCESSED_DATA_DIR / dataset_name / f"embeddings{suffix}"
        assert aug_embeddings_dir.is_dir(), f"Augment embeddings dir not found: {embeddings_dir}"
        aug_manifest_path = aug_embeddings_dir / "manifest.csv"
        assert aug_manifest_path.is_file(), f"Augment manifest file not found: {aug_manifest_path}"
        aug_manifest = pd.read_csv(aug_manifest_path)

    set_seed(args.seed)

    hyperparams = {
        "epochs": epochs,
        "patience": patience,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "loss_type": loss_type if prediction_mode == "VA" else "cross_entropy",
    }

    # ========================================================================
    # MERGE DATASET: Use predefined splits
    # ========================================================================
    if dataset_name == "MERGE":
        components = get_mode_components(prediction_mode)
        data = prepare_datasets_merge(dataset, merge_split, components, args.batch_size, aug_manifest, augment_size, augment)

        train_s = len(data["train_loader"].dataset)
        val_s = len(data["val_loader"].dataset)
        test_s = len(data["test_loader"].dataset)
        logger.info(
            f"MERGE split {merge_split}: {train_s} train, {val_s} val, {test_s} test"
        )

        model = components["model_class"](
            in_dim=data["train_loader"].dataset.input_dim, hidden_dim=hidden_dim, dropout=dropout
        ).to(device)

        dummy_input = torch.randn(1, 1, data["train_loader"].dataset.input_dim).to(device)
        writer.add_graph(model, dummy_input)

        if prediction_mode == "VA":
            model, score = components["train_fn"](
                name="merge_va",
                model=model,
                args=args,
                train_loader=data["train_loader"],
                test_loader=data["val_loader"],
                report_dir=report_dir,
                writer=writer,
                scatter=True,
            )
            test_score = components["eval_fn"](model, data["test_loader"], device, writer, report_dir, scatter=True)
        else:
            model, score = components["train_fn"](
                name="merge_russell4q",
                model=model,
                args=args,
                train_loader=data["train_loader"],
                test_loader=data["val_loader"],
                report_dir=report_dir,
                writer=writer,
                confusion=True,
            )
            test_score = components["eval_fn"](model, data["test_loader"], device, writer, report_dir, confusion=True)
        
        model_path = report_dir / "model.pth"
        torch.save(model, model_path)

        metric_name = components["metric_name"]
        save_training_summary(
            report_dir / "training_summary.json",
            model_path=model_path,
            dataset_name=dataset_name,
            prediction_mode=prediction_mode,
            train_size=train_s,
            validation_size=val_s,
            test_size=test_s,
            validation_score=score[metric_name],
            test_score=test_score[metric_name],
            merge_split=merge_split,
            hyperparameters=hyperparams
        )

    # ========================================================================
    # DEAM/PMEmo: Use K-fold cross-validation
    # ========================================================================
    else:
        manifest_path = PROCESSED_DATA_DIR / dataset_name / "embeddings" / "manifest.csv"
        assert manifest_path.is_file(), f"Manifest file not found: {manifest_path}"
        manifest = pd.read_csv(manifest_path)
        components = get_mode_components(prediction_mode)
        split_data = prepare_kfold(manifest, args.test_size, kfolds, seed)

        save_splits(
            report_dir / "splits.json",
            dataset_name=dataset_name,
            prediction_mode=prediction_mode,
            folds=split_data["folds"],
            head=head,
            hyperparameters=hyperparams,
            seed=seed
        )

        logger.info(f"Training started. Report dir: {report_dir}")

        # K-fold validation
        fold_scores = []

        kfolds = prepare_kfold_dataset(manifest, dataset, split_data["folds"], components, batch_size, labels_scale, aug_manifest, augment_size)

        for i, train_loader, valid_loader in kfolds["folds"]:
            logger.info(f"Training fold {i}...")
            logger.info(
                f"Fold {i} size: {len(train_loader.dataset)} train, {len(valid_loader.dataset)} validation"
            )

            model = components["model_class"](in_dim=kfolds["dataset"].input_dim, hidden_dim=hidden_dim, dropout=dropout).to(
                device
            )

            if i == 1:
                dummy_input = torch.randn(1, 1, kfolds["dataset"].input_dim).to(device)
                writer.add_graph(model, dummy_input)

            _, score = components["train_fn"](
                name=f"fold_{i}",
                model=model,
                args=args,
                train_loader=train_loader,
                test_loader=valid_loader,
                report_dir=report_dir,
                writer=writer,
            )
            fold_scores.append(score[components["metric_name"]])

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        logger.info(f"K-fold validation {components["metric_name"]}: {avg_score:.4f} ± {std_score:.4f}")

        logger.info("Training final model on full training set...")

        data = prepare_datasets(dataset, manifest, split_data["train_indices"], split_data["test_indices"], components, batch_size, labels_scale, aug_manifest, augment_size)
        logger.info(f"Final size: {len(data["train_loader"].dataset)} train, {len(data["test_loader"].dataset)} test")

        model = components["model_class"](in_dim=data["dataset"].input_dim, hidden_dim=hidden_dim, dropout=dropout).to(
            device
        )

        # Prepare kwargs based on prediction mode
        train_kwargs = {
            "name": "final",
            "model": model,
            "args": args,
            "train_loader": data["train_loader"],
            "test_loader": data["test_loader"],
            "report_dir": report_dir,
            "writer": writer,
        }

        # Add mode-specific argument
        if prediction_mode == "VA":
            train_kwargs["scatter"] = True
        else:  # Russell4Q
            train_kwargs["confusion"] = True

        model, score = components["train_fn"](**train_kwargs)
        model_path = report_dir / "model.pth"
        torch.save(model, model_path)

        table = "| Metric | Value |\n|:--|--:|\n"
        for key, value in score.items():
            if key != "Confusion_Matrix":
                table += f"| {key} | {value:.4f} |\n"

        writer.add_text("Test results", table)

        save_training_summary(
            report_dir / "training_summary.json",
            model_path=model_path,
            dataset_name=dataset_name,
            prediction_mode=prediction_mode,
            train_size=len(data["train_loader"].dataset),
            test_size=len(data["test_loader"].dataset),
            test_score=score[components["metric_name"]],
            kfold_mean=avg_score,
            kfold_std=std_score,
            hyperparameters=hyperparams
        )

    writer.close()
    logger.success("Model training complete.")


if __name__ == "__main__":
    app()
