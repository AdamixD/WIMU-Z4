import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Subset, ConcatDataset

from mer.datasets.common import (
    SongSequenceDataset,
    SongClassificationDataset,
)
from mer.modeling.utils.data_loaders import (
    build_items_merge_regression,
    build_items_merge_classification,
    build_items_regression,
    build_items_classification
)

from mer.heads import BiGRUHead, BiGRUClassificationHead
from mer.modeling.utils.misc import pad_and_mask, pad_and_mask_classification

def prepare_kfold(manifest, test_size=0.2, kfolds=5, seed=42):
    n_samples = len(manifest)
    indices = np.arange(n_samples)
    
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True
    )
    
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(train_indices):
        folds.append(
            {"train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist()}
        )
    
    return {
        "folds": folds,
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist()
    }

def get_mode_components(prediction_mode):
    from mer.modeling.train import (
        train_model_regression,
        train_model_classification,
    )
    from mer.modeling.train import (
        evaluate_model_regression,
        evaluate_model_classification,
    )
    
    if prediction_mode == "VA":
        return {
            "mode": "VA",
            "build_items_df": build_items_merge_regression,     # for MERGE
            "build_items_manifest": build_items_regression,     # for K-Fold
            "dataset_class": SongSequenceDataset,
            "collate_fn": pad_and_mask,
            "model_class": BiGRUHead,
            "train_fn": train_model_regression,
            "eval_fn": evaluate_model_regression,
            "metric_name": "CCC_mean",
        }

    else:  # Russell4Q
        return {
            "mode": "Russell4Q",
            "build_items_df": build_items_merge_classification,     # for MERGE    
            "build_items_manifest": build_items_classification,     # for K-Fold
            "dataset_class": SongClassificationDataset,
            "collate_fn": pad_and_mask_classification,
            "model_class": BiGRUClassificationHead,
            "train_fn": train_model_classification,
            "eval_fn": evaluate_model_classification,
            "metric_name": "Accuracy",
        }

def create_loader(items, config, batch_size, shuffle=False):
    dataset = config["dataset_class"](items)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=config["collate_fn"],
        shuffle=shuffle,
    )

def prepare_datasets_merge(dataset, merge_split, components, batch_size, augment_manifest=None, augment_size=0.3, augment_name=None):
    train_df, val_df, test_df = dataset.load_train_val_test_splits(merge_split)
    train_items = components["build_items_df"](train_df, dataset)
    val_items   = components["build_items_df"](val_df, dataset)
    test_items  = components["build_items_df"](test_df, dataset)
    train_dset = components["dataset_class"](train_items)
    val_dset   = components["dataset_class"](val_items)
    test_dset  = components["dataset_class"](test_items)

    if augment_manifest is not None:
        train_song_ids = train_df['song_id'].tolist()
        subset_size = int(augment_size * len(train_song_ids))
        aug_song_ids = random.sample(train_song_ids, subset_size)
        aug_train = augment_manifest[augment_manifest['song_id'].isin(aug_song_ids)]
        augment_items = components["build_items_df"](aug_train, dataset, augment_name=augment_name)
        augment_dset = components["dataset_class"](augment_items)
        train_dset = ConcatDataset([train_dset, augment_dset])

    train_loader = create_loader(train_dset, components, batch_size, shuffle=True)
    val_loader   = create_loader(val_dset,   components, batch_size)
    test_loader  = create_loader(test_dset,  components, batch_size)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader
    }

def prepare_kfold_dataset(manifest, dataset, folds, components, batch_size, labels_scale=None, augment_manifest=None, augment_size=0.3):
    items = components["build_items_manifest"](manifest=manifest, dataset=dataset, labels_scale=labels_scale)
    dset = components["dataset_class"](items)
    if augment_manifest is not None:
        augment_items = components["build_items_manifest"](manifest=augment_manifest, dataset=dataset, labels_scale=labels_scale)
        augment_dset = components["dataset_class"](augment_items)

    def fold_generator():
        for idx, fold in enumerate(folds, start=1):

            train_subset = Subset(dset, fold["train_indices"])
            if augment_manifest is not None:
                train_indices = fold["train_indices"]
                subset_size = int(augment_size * len(train_indices))
                aug_indices = random.sample(train_indices, subset_size)
                aug_train = Subset(augment_dset, aug_indices)
                train_subset = ConcatDataset([train_subset, aug_train])

            val_subset   = Subset(dset, fold["validation_indices"])

            train_loader = create_loader(train_subset, components, batch_size, True)
            val_loader = create_loader(val_subset, components, batch_size)

            yield idx, train_loader, val_loader

    return {
        "dataset": dset,
        "folds": fold_generator()
    }

def prepare_datasets(dataset, manifest, train_indices, test_indices, components, batch_size, labels_scale=None, augment_manifest=None, augment_size=0.3):

    items = components["build_items_manifest"](manifest=manifest, dataset=dataset, labels_scale=labels_scale)
    dset = components["dataset_class"](items)
    train_subset = Subset(dset, indices=train_indices)
    test_subset = Subset(dset, indices=test_indices)
    if augment_manifest is not None:
        augment_items = components["build_items_manifest"](manifest=augment_manifest, dataset=dataset, labels_scale=labels_scale)
        augment_dset = components["dataset_class"](augment_items)
        subset_size = int(augment_size * len(train_indices))
        aug_indices = random.sample(train_indices, subset_size)
        aug_train = Subset(augment_dset, aug_indices)
        train_subset = ConcatDataset([train_subset, aug_train])
    

    train_loader = create_loader(train_subset, components, batch_size, shuffle=True)
    test_loader  = create_loader(test_subset,  components, batch_size)

    return {
        "dataset": dset,
        "train_loader": train_loader,
        "test_loader": test_loader
    }