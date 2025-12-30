"""Compare VA-to-Russell4Q vs Direct Russell4Q classification models."""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from mer.config import PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR
from mer.datasets import DEAMDataset, MERGEDataset, PMEmoDataset
from mer.datasets.common import SongClassificationDataset, SongSequenceDataset
from mer.modeling.utils.data_loaders import (
    build_items_classification,
    build_items_merge_classification,
    build_items_merge_regression,
    build_items_regression,
)
from mer.modeling.utils.metrics import classification_metrics, va_to_russell4q
from mer.modeling.utils.misc import pad_and_mask, pad_and_mask_classification, set_seed


def evaluate_va_model_as_russell4q(model, test_loader, device):
    """
    Evaluate a VA regression model by mapping predictions to Russell 4Q.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, Y, mask in tqdm(test_loader, desc="Evaluating VA→Russell4Q", leave=False):
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)

            pred = model(X)  # (batch_size, seq_len, 2)

            valid = mask > 0
            if valid.sum() == 0:
                continue

            # Extract valid predictions and targets
            pred_v = pred[..., 0][valid].cpu().numpy()
            pred_a = pred[..., 1][valid].cpu().numpy()
            true_v = Y[..., 0][valid].cpu().numpy()
            true_a = Y[..., 1][valid].cpu().numpy()

            # Map to Russell 4Q
            pred_quadrants = va_to_russell4q(pred_v, pred_a)
            true_quadrants = va_to_russell4q(true_v, true_a)

            all_preds.extend(pred_quadrants.tolist())
            all_targets.extend(true_quadrants.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = classification_metrics(all_targets, all_preds)
    return metrics


def evaluate_russell4q_model(model, test_loader, device):
    """
    Evaluate a Russell 4Q classification model directly.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, Y, mask in tqdm(test_loader, desc="Evaluating Russell4Q", leave=False):
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)

            logits = model(X)  # (batch_size, seq_len, 4)
            pred = torch.argmax(logits, dim=-1)

            valid = mask > 0
            if valid.sum() == 0:
                continue

            all_preds.extend(pred[valid].cpu().numpy().tolist())
            all_targets.extend(Y[valid].cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = classification_metrics(all_targets, all_preds)
    return metrics


def main(
    va_model: str = typer.Option(..., help="Path to VA regression model (.pth)"),
    russell4q_model: str = typer.Option(..., help="Path to Russell4Q classification model (.pth)"),
    dataset_name: str = typer.Option(
        ..., help="Dataset to use for comparison (DEAM, PMEmo, MERGE)"
    ),
    labels_scale: str = typer.Option("norm", help="Labels scale (norm or 19)"),
    batch_size: int = typer.Option(8, help="Batch size for evaluation"),
    seed: int = typer.Option(42, help="Random seed"),
    test_size: float = typer.Option(0.1, help="Test split size (for DEAM/PMEmo)"),
    merge_split: str = typer.Option("70_15_15", help="MERGE split to use (70_15_15 or 80_10_10)"),
    output_dir: str = typer.Option(None, help="Output directory for comparison report"),
):
    """Compare VA-to-Russell4Q mapping vs Direct Russell4Q classification."""
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Convert paths
    va_model_path = Path(va_model)
    russell4q_model_path = Path(russell4q_model)

    # Load models
    logger.info(f"Loading VA model from {va_model_path}")
    va_model_obj = torch.load(va_model_path, map_location=device)
    va_model_obj.eval()

    logger.info(f"Loading Russell4Q model from {russell4q_model_path}")
    russell4q_model_obj = torch.load(russell4q_model_path, map_location=device)
    russell4q_model_obj.eval()

    # Load dataset
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
        dataset = MERGEDataset(
            root_dir=RAW_DATA_DIR / dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR / dataset_name / "embeddings",
        )

    # Prepare test data
    if dataset_name == "MERGE":
        # Use predefined test split
        train_manifest, val_manifest, test_manifest = dataset.load_train_val_test_splits(
            merge_split
        )

        # Build items for VA and Russell4Q
        va_items = build_items_regression(test_manifest, dataset, labels_scale=labels_scale)
        russell4q_items = build_items_classification(
            test_manifest, dataset, labels_scale=labels_scale
        )

        va_test_dataset = SongSequenceDataset(va_items)
        russell4q_test_dataset = SongClassificationDataset(russell4q_items)

    else:
        # Use K-fold approach for DEAM/PMEmo
        manifest_path = PROCESSED_DATA_DIR / dataset_name / "embeddings" / "manifest.csv"
        manifest = pd.read_csv(manifest_path)

        from sklearn.model_selection import train_test_split

        n_samples = len(manifest)
        indices = np.arange(n_samples)
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=seed, shuffle=True
        )

        test_manifest = manifest.iloc[test_indices].reset_index(drop=True)

        # Build items for VA and Russell4Q
        va_items = build_items_regression(test_manifest, dataset, labels_scale=labels_scale)
        russell4q_items = build_items_classification(
            test_manifest, dataset, labels_scale=labels_scale
        )

        va_test_dataset = SongSequenceDataset(va_items)
        russell4q_test_dataset = SongClassificationDataset(russell4q_items)

    # Create data loaders
    va_test_loader = DataLoader(
        va_test_dataset, batch_size=batch_size, collate_fn=pad_and_mask, shuffle=False,
    )
    russell4q_test_loader = DataLoader(
        russell4q_test_dataset,
        batch_size=batch_size,
        collate_fn=pad_and_mask_classification,
        shuffle=False,
    )

    logger.info("Evaluating models...")

    # Evaluate VA model (mapped to Russell4Q)
    va_metrics = evaluate_va_model_as_russell4q(va_model_obj, va_test_loader, device)

    # Evaluate Russell4Q model (direct classification)
    russell4q_metrics = evaluate_russell4q_model(
        russell4q_model_obj, russell4q_test_loader, device
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)

    logger.info("\n VA Model → Russell4Q Mapping:")
    logger.info(f"  Accuracy:  {va_metrics['Accuracy']:.4f}")
    logger.info(f"  Precision: {va_metrics['Precision']:.4f}")
    logger.info(f"  Recall:    {va_metrics['Recall']:.4f}")
    logger.info(f"  F1 Score:  {va_metrics['F1']:.4f}")

    logger.info("\n Direct Russell4Q Classification:")
    logger.info(f"  Accuracy:  {russell4q_metrics['Accuracy']:.4f}")
    logger.info(f"  Precision: {russell4q_metrics['Precision']:.4f}")
    logger.info(f"  Recall:    {russell4q_metrics['Recall']:.4f}")
    logger.info(f"  F1 Score:  {russell4q_metrics['F1']:.4f}")

    logger.info("\n Difference (Russell4Q - VA→Russell4Q):")
    logger.info(f"  Accuracy:  {russell4q_metrics['Accuracy'] - va_metrics['Accuracy']:+.4f}")
    logger.info(f"  Precision: {russell4q_metrics['Precision'] - va_metrics['Precision']:+.4f}")
    logger.info(f"  Recall:    {russell4q_metrics['Recall'] - va_metrics['Recall']:+.4f}")
    logger.info(f"  F1 Score:  {russell4q_metrics['F1'] - va_metrics['F1']:+.4f}")

    logger.info("=" * 80)

    # Save results
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = PROJ_ROOT / "reports" / f"comparison_{dataset_name}"

    output_path.mkdir(parents=True, exist_ok=True)

    comparison_report = {
        "dataset": dataset_name,
        "va_model": str(va_model_path),
        "russell4q_model": str(russell4q_model_path),
        "test_samples": len(va_test_dataset),
        "va_mapped": {
            "Accuracy": float(va_metrics["Accuracy"]),
            "Precision": float(va_metrics["Precision"]),
            "Recall": float(va_metrics["Recall"]),
            "F1": float(va_metrics["F1"]),
            "Confusion_Matrix": va_metrics["Confusion_Matrix"].tolist(),
        },
        "russell4q_direct": {
            "Accuracy": float(russell4q_metrics["Accuracy"]),
            "Precision": float(russell4q_metrics["Precision"]),
            "Recall": float(russell4q_metrics["Recall"]),
            "F1": float(russell4q_metrics["F1"]),
            "Confusion_Matrix": russell4q_metrics["Confusion_Matrix"].tolist(),
        },
        "difference": {
            "Accuracy": float(russell4q_metrics["Accuracy"] - va_metrics["Accuracy"]),
            "Precision": float(russell4q_metrics["Precision"] - va_metrics["Precision"]),
            "Recall": float(russell4q_metrics["Recall"] - va_metrics["Recall"]),
            "F1": float(russell4q_metrics["F1"] - va_metrics["F1"]),
        },
    }

    report_path = output_path / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison_report, f, indent=2)

    logger.info(f"\nComparison report saved to: {report_path}")


if __name__ == "__main__":
    typer.run(main)
