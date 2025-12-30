"""
Evaluation script for trained models with support for both VA regression and Russell 4Q classification.
"""

from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
import typer

from mer.config import DEFAULT_DEVICE, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mer.datasets.common import SongClassificationDataset, SongSequenceDataset
from mer.datasets.deam import DEAMDataset
from mer.datasets.merge import MERGEDataset
from mer.datasets.pmemo import PMEmoDataset
from mer.modeling.utils.data_loaders import (
    build_items_classification,
    build_items_merge_classification,
    build_items_merge_regression,
    build_items_regression,
)
from mer.modeling.utils.metrics import (
    classification_metrics,
    metrics_dict,
    quadrant_to_name,
    va_to_russell_quadrant,
)
from mer.modeling.utils.misc import pad_and_mask, pad_and_mask_classification

app = typer.Typer()


def evaluate_regression(
    model, dataloader, device, output_dir: Path = None, map_to_russell4q: bool = False
):
    """
    Evaluate VA regression model.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        output_dir: Directory to save results
        map_to_russell4q: If True, also evaluate VA→Russell4Q mapping
    
    Returns:
        Dictionary with metrics: CCC, Pearson, R2, RMSE for both valence and arousal
    """
    Ys = []
    Ps = []

    model.eval()
    with torch.no_grad():
        for Xb, Yb, Mb in dataloader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Mb = Mb.to(device)
            P = model(Xb)
            M = Mb.unsqueeze(-1)
            Ys.append((Yb * M).reshape(-1, 2)[Mb.reshape(-1) > 0].cpu().numpy())
            Ps.append((P * M).reshape(-1, 2)[Mb.reshape(-1) > 0].cpu().numpy())

    Y = np.concatenate(Ys, 0)
    P = np.concatenate(Ps, 0)

    # Calculate metrics
    vccc, vp, vr2, vrmse = metrics_dict(Y[:, 0], P[:, 0]).values()
    accc, ap, ar2, armse = metrics_dict(Y[:, 1], P[:, 1]).values()

    results = {
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

    # Add Russell 4Q mapping evaluation if requested
    if map_to_russell4q:
        Y_quadrants = va_to_russell_quadrant(Y[:, 0], Y[:, 1])
        P_quadrants = va_to_russell_quadrant(P[:, 0], P[:, 1])

        russell4q_metrics = classification_metrics(Y_quadrants, P_quadrants)
        results.update({f"Russell4Q_{k}": v for k, v in russell4q_metrics.items()})

        logger.info("VA→Russell 4Q Mapping Evaluation:")
        for k, v in russell4q_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # Create visualizations if output_dir is provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scatter plots for VA
        for i, dim in enumerate(["valence", "arousal"]):
            plt.figure(figsize=(6, 6))
            plt.scatter(Y[:, i], P[:, i], s=8, alpha=0.4)
            lo = float(min(Y[:, i].min(), P[:, i].min()))
            hi = float(max(Y[:, i].max(), P[:, i].max()))
            plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
            plt.title(f"{dim.capitalize()}: Ground Truth vs Prediction")
            plt.xlabel("Ground Truth")
            plt.ylabel("Prediction")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f"scatter_{dim}.png", dpi=150)
            plt.close()

        # Save predictions
        df_data = {
            "valence_true": Y[:, 0],
            "valence_pred": P[:, 0],
            "arousal_true": Y[:, 1],
            "arousal_pred": P[:, 1],
        }

        if map_to_russell4q:
            df_data["russell4q_true_class"] = Y_quadrants
            df_data["russell4q_pred_class"] = P_quadrants
            df_data["russell4q_true_name"] = quadrant_to_name(Y_quadrants)
            df_data["russell4q_pred_name"] = quadrant_to_name(P_quadrants)

            # Create confusion matrix for Russell 4Q
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(Y_quadrants, P_quadrants, labels=[0, 1, 2, 3])

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            quadrant_labels = ["Q1\n(Happy)", "Q2\n(Angry)", "Q3\n(Sad)", "Q4\n(Calm)"]
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=quadrant_labels,
                yticklabels=quadrant_labels,
                title="VA→Russell 4Q Mapping - Confusion Matrix",
                ylabel="True Quadrant (from Ground Truth VA)",
                xlabel="Predicted Quadrant (from Predicted VA)",
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
                        fontsize=14,
                    )

            fig.tight_layout()
            plt.savefig(output_dir / "confusion_matrix_va_to_russell4q.png", dpi=150)
            plt.close()

        df = pd.DataFrame(df_data)
        df.to_csv(output_dir / "predictions.csv", index=False)

    return results


def evaluate_classification(model, dataloader, device, output_dir: Path = None):
    """
    Evaluate Russell 4Q classification model.
    
    Returns metrics: Accuracy, Precision, Recall, F1
    """
    Ys = []
    Ps = []

    model.eval()
    with torch.no_grad():
        for Xb, Yb, Mb in dataloader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Mb = Mb.to(device)

            # Get logits and convert to predictions
            logits = model(Xb)
            pred = torch.argmax(logits, dim=-1)

            # Apply mask
            valid = Mb > 0
            Ys.append(Yb[valid].cpu().numpy())
            Ps.append(pred[valid].cpu().numpy())

    Y = np.concatenate(Ys, 0)
    P = np.concatenate(Ps, 0)

    results = classification_metrics(Y, P)

    # Create confusion matrix and classification report if output_dir is provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        cm = confusion_matrix(Y, P, labels=[0, 1, 2, 3])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        quadrant_labels = [
            "Q1\n(Happy/Excited)",
            "Q2\n(Angry/Tense)",
            "Q3\n(Sad/Depressed)",
            "Q4\n(Calm/Relaxed)",
        ]
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=quadrant_labels,
            yticklabels=quadrant_labels,
            title="Confusion Matrix - Russell 4Q Classification",
            ylabel="True Quadrant",
            xlabel="Predicted Quadrant",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
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
                    fontsize=14,
                )

        fig.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # Classification report
        report = classification_report(Y, P, target_names=quadrant_labels, digits=4)
        with open(output_dir / "classification_report.txt", "w") as f:
            f.write(report)

        # Save predictions
        df = pd.DataFrame(
            {
                "true_class": Y,
                "predicted_class": P,
                "true_quadrant": [f"Q{y+1}" for y in Y],
                "predicted_quadrant": [f"Q{p+1}" for p in P],
            }
        )
        df.to_csv(output_dir / "predictions.csv", index=False)

    return results


@app.command()
def main(
    model_path: Annotated[Path, typer.Argument(help="Path to trained model (.pth file)")],
    dataset_name: Annotated[
        Literal["DEAM", "PMEmo", "MERGE"], typer.Option(case_sensitive=False)
    ] = "MERGE",
    prediction_mode: Annotated[
        Literal["VA", "Russell4Q"],
        typer.Option(case_sensitive=False, help="VA for regression, Russell4Q for classification"),
    ] = "VA",
    split: Annotated[
        Literal["train", "val", "test"],
        typer.Option(case_sensitive=False, help="Which split to evaluate"),
    ] = "test",
    merge_split: Annotated[
        Literal["70_15_15", "40_30_30"],
        typer.Option(case_sensitive=False, help="MERGE dataset split ratio"),
    ] = "70_15_15",
    batch_size: int = 6,
    labels_scale: Annotated[
        Literal["19", "norm"],
        typer.Option(case_sensitive=False, help="Scale of dynamic labels in DEAM source files"),
    ] = "norm",
    test_size: float = typer.Option(
        0.1, help="Test set size for DEAM/PMEmo (used with --seed to replicate train/test split)"
    ),
    seed: int = typer.Option(
        42, help="Random seed for DEAM/PMEmo split (should match training seed)"
    ),
    output_dir: Annotated[Path, typer.Option(help="Directory to save evaluation results")] = None,
    map_to_russell4q: bool = typer.Option(
        False, "--map-to-russell4q", help="For VA models: map predictions to Russell 4Q quadrants"
    ),
    device: Annotated[Literal["cuda", "cpu"], typer.Option(case_sensitive=False)] = DEFAULT_DEVICE,
):
    """
    Evaluate a trained model on a dataset.
    
    Examples:
        # Evaluate VA model on MERGE test set
        python -m mer.modeling.evaluate model.pth --dataset-name MERGE --prediction-mode VA --split test
        
        # Evaluate Russell 4Q model on MERGE test set
        python -m mer.modeling.evaluate model.pth --dataset-name MERGE --prediction-mode Russell4Q --split test
        
        # Evaluate DEAM VA model on test set (must match training seed/test-size)
        python -m mer.modeling.evaluate model.pth --dataset-name DEAM --prediction-mode VA --split test --test-size 0.1 --seed 42
        
        # Evaluate PMEmo VA model with Russell 4Q mapping
        python -m mer.modeling.evaluate model.pth --dataset-name PMEmo --prediction-mode VA --split test --test-size 0.1 --seed 42 --map-to-russell4q
    """

    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model first to detect its actual type
    logger.info("Loading model...")
    model = torch.load(model_path, map_location=device)
    model.eval()

    # Detect model type from architecture
    out_features = model.out.out_features
    actual_model_type = "VA" if out_features == 2 else "Russell4Q"
    logger.info(f"Detected model type: {actual_model_type} (out_features={out_features})")

    # Override prediction_mode with actual model type
    if prediction_mode != actual_model_type:
        logger.warning(
            f"Requested prediction_mode '{prediction_mode}' does not match model type '{actual_model_type}'. Using '{actual_model_type}'."
        )
        prediction_mode = actual_model_type

    if dataset_name == "MERGE" and prediction_mode not in ["VA", "Russell4Q"]:
        raise ValueError("MERGE dataset requires prediction_mode to be 'VA' or 'Russell4Q'")

    # Set output directory
    if output_dir is None:
        output_dir = model_path.parent / f"evaluation_{split}_{prediction_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Evaluating {model_path} on {dataset_name} ({split} split, {prediction_mode} mode)"
    )
    logger.info(f"Output directory: {output_dir}")

    # Initialize dataset
    if dataset_name == "DEAM":
        import numpy as np
        from sklearn.model_selection import train_test_split

        dataset = DEAMDataset(
            root_dir=RAW_DATA_DIR / dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR / dataset_name / "embeddings",
        )
        manifest_path = PROCESSED_DATA_DIR / dataset_name / "embeddings" / "manifest.csv"
        manifest = pd.read_csv(manifest_path)

        # Replicate train/test split from training
        n_samples = len(manifest)
        indices = np.arange(n_samples)
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=seed, shuffle=True
        )

        # Use test split by default, or train/val based on --split parameter
        if split == "test":
            selected_indices = test_indices
        elif split == "train":
            selected_indices = train_indices
        else:  # val - use a portion of train for validation
            from sklearn.model_selection import train_test_split as split_again

            train_only, val_indices = split_again(
                train_indices, test_size=0.2, random_state=seed, shuffle=True
            )
            selected_indices = val_indices

        manifest = manifest.iloc[selected_indices].reset_index(drop=True)
        logger.info(
            f"Using {split} split with {len(manifest)} samples (test_size={test_size}, seed={seed})"
        )

        if prediction_mode == "VA":
            items = build_items_regression(manifest, dataset, labels_scale)
            dset = SongSequenceDataset(items)
            dataloader = DataLoader(dset, batch_size=batch_size, collate_fn=pad_and_mask)
            eval_fn = evaluate_regression
        else:  # Russell4Q
            items = build_items_classification(manifest, dataset, labels_scale)
            dset = SongClassificationDataset(items)
            dataloader = DataLoader(
                dset, batch_size=batch_size, collate_fn=pad_and_mask_classification
            )
            eval_fn = evaluate_classification

    elif dataset_name == "PMEmo":
        import numpy as np
        from sklearn.model_selection import train_test_split

        dataset = PMEmoDataset(
            root_dir=RAW_DATA_DIR / dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR / dataset_name / "embeddings",
        )
        manifest_path = PROCESSED_DATA_DIR / dataset_name / "embeddings" / "manifest.csv"
        manifest = pd.read_csv(manifest_path)

        # Replicate train/test split from training
        n_samples = len(manifest)
        indices = np.arange(n_samples)
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=seed, shuffle=True
        )

        # Use test split by default, or train/val based on --split parameter
        if split == "test":
            selected_indices = test_indices
        elif split == "train":
            selected_indices = train_indices
        else:  # val - use a portion of train for validation
            from sklearn.model_selection import train_test_split as split_again

            train_only, val_indices = split_again(
                train_indices, test_size=0.2, random_state=seed, shuffle=True
            )
            selected_indices = val_indices

        manifest = manifest.iloc[selected_indices].reset_index(drop=True)
        logger.info(
            f"Using {split} split with {len(manifest)} samples (test_size={test_size}, seed={seed})"
        )

        if prediction_mode == "VA":
            items = build_items_regression(manifest, dataset, labels_scale)
            dset = SongSequenceDataset(items)
            dataloader = DataLoader(dset, batch_size=batch_size, collate_fn=pad_and_mask)
            eval_fn = evaluate_regression
        else:  # Russell4Q
            items = build_items_classification(manifest, dataset, labels_scale)
            dset = SongClassificationDataset(items)
            dataloader = DataLoader(
                dset, batch_size=batch_size, collate_fn=pad_and_mask_classification
            )
            eval_fn = evaluate_classification

    elif dataset_name == "MERGE":
        dataset = MERGEDataset(
            root_dir=RAW_DATA_DIR / dataset_name,
            out_embeddings_dir=PROCESSED_DATA_DIR / dataset_name / "embeddings",
            mode=prediction_mode,
        )

        # Load appropriate split
        train_df, val_df, test_df = dataset.load_train_val_test_splits(merge_split)
        split_map = {"train": train_df, "val": val_df, "test": test_df}
        split_df = split_map[split]

        logger.info(f"Loaded {len(split_df)} samples from {split} split")

        if prediction_mode == "VA":
            items = build_items_merge_regression(split_df, dataset)
            dset = SongSequenceDataset(items)
            dataloader = DataLoader(dset, batch_size=batch_size, collate_fn=pad_and_mask)
            eval_fn = evaluate_regression
        else:  # Russell4Q
            items = build_items_merge_classification(split_df, dataset)
            dset = SongClassificationDataset(items)
            dataloader = DataLoader(
                dset, batch_size=batch_size, collate_fn=pad_and_mask_classification
            )
            eval_fn = evaluate_classification
    else:
        raise NotImplementedError(dataset_name)

    # Evaluate
    logger.info(f"Evaluating on {len(items)} samples...")
    if eval_fn == evaluate_regression:
        results = eval_fn(model, dataloader, device, output_dir, map_to_russell4q=map_to_russell4q)
    else:
        results = eval_fn(model, dataloader, device, output_dir)

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    for key, value in results.items():
        if key != "Confusion_Matrix":
            logger.info(f"{key:20s}: {value:.4f}")
    logger.info("=" * 50)

    # Save results to JSON
    import json

    results_json = {k: v.tolist() if k == "Confusion_Matrix" else v for k, v in results.items()}
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    logger.success(f"Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    app()
