import json
from pathlib import Path
from typing import List, Optional


def save_training_summary(
    report_path,
    model_path,
    dataset_name,
    prediction_mode,
    train_size,
    test_size,
    hyperparameters,
    validation_size=None,
    validation_score=None,
    merge_split=None,
    test_score=None,
    kfold_mean=None,
    kfold_std=None,
    augments: Optional[List[str]] = None,
    augment_size: Optional[float] = None,
):
    data = {
        "model_path": str(model_path),
        "dataset": dataset_name,
        "prediction_mode": prediction_mode,
        "training_size": train_size,
        "test_size": test_size,
    }

    if merge_split is not None:
        data["merge_split"] = merge_split
    if validation_size is not None:
        data["validation_size"] = validation_size
    if validation_score is not None:
        data["validation_score"] = validation_score
    if test_score is not None:
        data["test_score"] = test_score
    if kfold_mean is not None:
        data["kfold_validation_score_mean"] = kfold_mean
    if kfold_std is not None:
        data["kfold_validation_score_std"] = kfold_std
    if augments:
        data["augments"] = augments
        data["augment_size"] = augment_size

    data["hyperparameters"] = hyperparameters
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, indent=2))

def save_splits(
    report_path,
    dataset_name,
    prediction_mode,
    folds,
    head,
    hyperparameters,
    seed=None
):
    data = {
        "dataset_name": dataset_name,
        "prediction_mode": prediction_mode,
        "seed": seed,
        "folds": folds,
        "head": head,
        "train_params": hyperparameters
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, indent=2))


def save_optimization_summary(
    report_path,
    dataset_name: str,
    prediction_mode: str,
    best_value: float,
    best_params: dict,
    n_trials: int,
    head: str,
    seed: int,
    merge_split: Optional[str] = None,
):
    data = {
        "dataset_name": dataset_name,
        "prediction_mode": prediction_mode,
        "head": head,
        "seed": seed,
        "n_trials": n_trials,
        "best_value": best_value,
        "best_params": best_params,
    }

    if merge_split is not None:
        data["merge_split"] = merge_split

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, indent=2))