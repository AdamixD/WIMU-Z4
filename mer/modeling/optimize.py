from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal, Optional
from random import randint

from loguru import logger
import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import typer

from mer.config import DEFAULT_DEVICE, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from mer.datasets.deam import DEAMDataset
from mer.datasets.pmemo import PMEmoDataset
from mer.datasets.merge import MERGEDataset
from mer.modeling.utils.misc import set_seed
from mer.modeling.utils.train_utils import (
    prepare_kfold,
    prepare_kfold_dataset,
    prepare_datasets,
    prepare_datasets_merge,
    get_mode_components
)
from mer.modeling.utils.report import save_optimization_summary, save_splits, save_training_summary


app = typer.Typer()


def make_objective(dataset_comp, base_args, study_dir: Path, head_name):

    if dataset_comp["dataset_name"] != "MERGE":
        manifest_path = PROCESSED_DATA_DIR / dataset_comp["dataset_name"] / "embeddings" / "manifest.csv"
        assert manifest_path.is_file(), f"Manifest file not found: {manifest_path}"
        manifest = pd.read_csv(manifest_path)
        split_data = prepare_kfold(manifest, base_args.test_size, base_args.kfolds, base_args.seed)
        save_splits(
            study_dir / "splits.json",
            dataset_name=dataset_comp["dataset_name"],
            prediction_mode=dataset_comp["prediction_mode"],
            folds=split_data["folds"],
            head=base_args.head,
            hyperparameters=None,
            seed=base_args.seed
        )

    def objective(trial: optuna.trial.Trial):

        # Build search space depending on whether user provided param or not
        hparams = {}

        if base_args.lr is None:
            hparams["lr"] = trial.suggest_float("lr", 1e-5, 1e-4, log=True)

        if base_args.hidden_dim is None:
            hparams["hidden_dim"] = trial.suggest_int("hidden_dim", 64, 128, step=64)

        if base_args.dropout is None:
            hparams["dropout"] = trial.suggest_float("dropout", 0.1, 0.4)

        if base_args.batch_size is None:
            hparams["batch_size"] = trial.suggest_categorical("batch_size", [4, 6])

        merged = vars(base_args).copy()
        merged.update(hparams)
        args = SimpleNamespace(**merged)

        trial_dir = study_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=str(trial_dir / "tensorboard"))

        if dataset_comp["dataset_name"] == "MERGE":
            components = get_mode_components(dataset_comp["prediction_mode"], head_name)
            data = prepare_datasets_merge(dataset_comp["dataset"], dataset_comp["merge_split"], components, args.batch_size)
            model = components["model_class"](
                in_dim=data["train_loader"].dataset.input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout
            ).to(args.device)
            model, score = components["train_fn"](
                name=f"trial_{trial.number}",
                model=model,
                args=args,
                train_loader=data["train_loader"],
                test_loader=data["val_loader"],
                report_dir=trial_dir,
                writer=writer,
            )
            val_score = score[components["metric_name"]]
        else:
            components = get_mode_components(dataset_comp["prediction_mode"], head_name)
            fold_scores = []
            kfolds = prepare_kfold_dataset(manifest, dataset_comp["dataset"], split_data["folds"], components, args.batch_size, args.labels_scale)
            for i, train_loader, valid_loader in kfolds["folds"]:
                model = components["model_class"](
                    in_dim=kfolds["dataset"].input_dim, 
                    hidden_dim=args.hidden_dim, 
                    dropout=args.dropout).to(args.device)
                _, score = components["train_fn"](
                    name=f"trial_{trial.number}_fold{i}",
                    model=model,
                    args=args,
                    train_loader=train_loader,
                    test_loader=valid_loader,
                    report_dir=trial_dir,
                    writer=writer,
                )
                fold_scores.append(score[components["metric_name"]])

                trial.report(np.mean(fold_scores), step=i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            val_score = np.mean(fold_scores)

        metric_name = components["metric_name"]
        writer.add_hparams(
            hparam_dict=hparams,
            metric_dict={metric_name: float(val_score)},
        )
        writer.close()

        return val_score

    return objective


@app.command()
def run(
    dataset_name: Annotated[
        Literal["DEAM", "PMEmo", "MERGE"], typer.Option(case_sensitive=False)
    ] = "DEAM",
    prediction_mode: Annotated[
        Literal["VA", "Russell4Q"],
        typer.Option(case_sensitive=False, help="VA for regression, Russell4Q for classification"),
    ] = "VA",
    head: Annotated[
        Literal["BiGRU", "CNNLSTM"], typer.Option(case_sensitive=False)
    ] = "BiGRU",
    n_trials: int = 3,
    seed: int = randint(1, 1000000),
    test_size: float = 0.2,
    epochs: int = 5,
    patience: int = 15,
    kfolds: int = 5,
    device: Annotated[Literal["cuda", "cpu"], typer.Option(case_sensitive=False)] = DEFAULT_DEVICE,
    labels_scale: Annotated[
        Literal["19", "norm"],
        typer.Option(case_sensitive=False, help="Scale of dynamic labels in DEAM source files"),
    ] = "norm",
    merge_split: Annotated[
        Literal["70_15_15", "40_30_30"],
        typer.Option(case_sensitive=False, help="MERGE dataset split ratio"),
    ] = "70_15_15",
    loss_type: Annotated[
        Literal["ccc", "mse", "hybrid"], typer.Option(case_sensitive=False)
    ] = "ccc",
    lr: Optional[float] = None,
    hidden_dim: Optional[int] = None,
    dropout: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    set_seed(seed)

    # Validate prediction mode
    if prediction_mode not in ["VA", "Russell4Q"]:
        raise ValueError("prediction_mode must be 'VA' or 'Russell4Q'")

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

    study_dir = REPORTS_DIR / f"optimize_{dataset_name}_{prediction_mode}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    study_dir.mkdir(parents=True)
    logger.info(f"Study directory: {study_dir}")

    base_args = SimpleNamespace(
        epochs=epochs,
        patience=patience,
        device=device,
        head=head,
        labels_scale=labels_scale,
        seed=seed,
        kfolds=kfolds,
        test_size=test_size,
        loss_type=loss_type,
        lr=lr,
        hidden_dim=hidden_dim,
        dropout=dropout,
        batch_size=batch_size,
    )

    dataset_comp = {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "prediction_mode": prediction_mode,
    }

    if dataset_name == "MERGE":
        dataset_comp["merge_split"] = merge_split

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(
        dataset_comp=dataset_comp,
        base_args=base_args,
        study_dir=study_dir,
        head_name=head
    )

    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_params = best.params
    
    metric_name = "CCC_mean" if prediction_mode == "VA" else "F1"
    logger.success(f"Best {metric_name} = {best.value:.4f}")
    logger.success(f"Best params = {best_params}")

    logger.info("Training final model with best hyperparameters...")
    
    final_dir = study_dir / "best_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    final_args = SimpleNamespace(
        epochs=epochs,
        patience=patience,
        device=device,
        lr=best_params.get("lr", lr or 1e-4),
        hidden_dim=best_params.get("hidden_dim", hidden_dim or 128),
        dropout=best_params.get("dropout", dropout or 0.2),
        batch_size=best_params.get("batch_size", batch_size or 6),
        loss_type=loss_type,
    )
    
    writer = SummaryWriter(log_dir=str(final_dir / "tensorboard"))
    components = get_mode_components(prediction_mode, head)
    
    if dataset_name == "MERGE":
        data = prepare_datasets_merge(dataset, merge_split, components, final_args.batch_size)
        
        model = components["model_class"](
            in_dim=data["train_loader"].dataset.input_dim,
            hidden_dim=final_args.hidden_dim,
            dropout=final_args.dropout
        ).to(device)
        
        # Add model graph to tensorboard
        dummy_input = torch.randn(1, 1, data["train_loader"].dataset.input_dim).to(device)
        writer.add_graph(model, dummy_input)
        
        train_kwargs = {
            "name": "final",
            "model": model,
            "args": final_args,
            "train_loader": data["train_loader"],
            "test_loader": data["val_loader"],
            "report_dir": final_dir,
            "writer": writer,
        }
        
        if prediction_mode == "VA":
            train_kwargs["scatter"] = True
        else:
            train_kwargs["confusion"] = True
        
        model, val_score = components["train_fn"](**train_kwargs)
        
        # Evaluate on test set
        if prediction_mode == "VA":
            test_score = components["eval_fn"](model, data["test_loader"], device, writer, final_dir, scatter=True)
        else:
            test_score = components["eval_fn"](model, data["test_loader"], device, writer, final_dir, confusion=True)
        
        train_size = len(data["train_loader"].dataset)
        val_size = len(data["val_loader"].dataset)
        test_size_final = len(data["test_loader"].dataset)
        
        save_training_summary(
            final_dir / "training_summary.json",
            model_path=final_dir / "model.pth",
            dataset_name=dataset_name,
            prediction_mode=prediction_mode,
            train_size=train_size,
            validation_size=val_size,
            test_size=test_size_final,
            validation_score=val_score[components["metric_name"]],
            test_score=test_score[components["metric_name"]],
            merge_split=merge_split,
            hyperparameters=best_params,
        )
        
    else:  # DEAM/PMEmo
        manifest_path = PROCESSED_DATA_DIR / dataset_name / "embeddings" / "manifest.csv"
        manifest = pd.read_csv(manifest_path)
        split_data = prepare_kfold(manifest, test_size, kfolds, seed)
        
        data = prepare_datasets(
            dataset, manifest, 
            split_data["train_indices"], 
            split_data["test_indices"], 
            components, 
            final_args.batch_size, 
            labels_scale
        )
        
        model = components["model_class"](
            in_dim=data["dataset"].input_dim,
            hidden_dim=final_args.hidden_dim,
            dropout=final_args.dropout
        ).to(device)
        
        # Add model graph to tensorboard
        dummy_input = torch.randn(1, 1, data["dataset"].input_dim).to(device)
        writer.add_graph(model, dummy_input)
        
        train_kwargs = {
            "name": "final",
            "model": model,
            "args": final_args,
            "train_loader": data["train_loader"],
            "test_loader": data["test_loader"],
            "report_dir": final_dir,
            "writer": writer,
        }
        
        if prediction_mode == "VA":
            train_kwargs["scatter"] = True
        else:
            train_kwargs["confusion"] = True
        
        model, test_score = components["train_fn"](**train_kwargs)
        
        train_size = len(data["train_loader"].dataset)
        test_size_final = len(data["test_loader"].dataset)
        
        save_training_summary(
            final_dir / "training_summary.json",
            model_path=final_dir / "model.pth",
            dataset_name=dataset_name,
            prediction_mode=prediction_mode,
            train_size=train_size,
            test_size=test_size_final,
            test_score=test_score[components["metric_name"]],
            hyperparameters=best_params,
        )
    
    # Save model
    model_path = final_dir / "model.pth"
    torch.save(model, model_path)
    
    writer.close()
    
    logger.success(f"Final model saved to {model_path}")
    logger.success(f"Final {metric_name} = {test_score[components['metric_name']]:.4f}")
    
    # Save optimization summary with final model info
    save_optimization_summary(
        report_path=study_dir / "best_hparams.json",
        dataset_name=dataset_name,
        prediction_mode=prediction_mode,
        best_value=best.value,
        best_params=best_params,
        n_trials=n_trials,
        head=head,
        seed=seed,
        merge_split=merge_split if dataset_name == "MERGE" else None,
    )


if __name__ == "__main__":
    app()
