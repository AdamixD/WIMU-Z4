"""Unified data loading utilities for training, evaluation, and comparison."""

import numpy as np
from tqdm import tqdm

from mer.modeling.utils.metrics import labels_convert, va_to_russell4q


def build_items_regression(manifest, dataset, labels_scale: str = "19"):
    """Build items for VA regression mode."""
    vmap, amap = dataset.va_maps
    items = []

    for r in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Preparing VA items",
        leave=False,
    ):
        sid = int(r.song_id)
        if not r.annotated:
            continue

        v, a = vmap[sid], amap[sid]
        L = min(len(v), len(a))
        if L <= 1:
            continue

        Y = np.stack([v[:L], a[:L]], axis=1).astype("float32")
        Y = labels_convert(Y, src=labels_scale, dst="norm").astype("float32")
        X = np.load(r.embeddings_path).astype("float32")

        T = min(len(X), len(Y))
        X, Y = X[:T], Y[:T]

        if len(X) == 0:
            continue

        items.append((X, Y))
    return items


def build_items_classification(manifest, dataset, labels_scale: str = "19"):
    """Build items for Russell4Q classification mode."""
    vmap, amap = dataset.va_maps
    items = []

    for r in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Preparing Russell4Q items",
        leave=False,
    ):
        sid = int(r.song_id)
        if not r.annotated:
            continue

        v, a = vmap[sid], amap[sid]
        L = min(len(v), len(a))
        if L <= 1:
            continue

        v_norm = labels_convert(v[:L], src=labels_scale, dst="norm").astype("float32")
        a_norm = labels_convert(a[:L], src=labels_scale, dst="norm").astype("float32")
        Y = va_to_russell4q(v_norm, a_norm).astype("int64")

        X = np.load(r.embeddings_path).astype("float32")

        T = min(len(X), len(Y))
        X, Y = X[:T], Y[:T]

        if len(X) == 0:
            continue

        items.append((X, Y))
    return items


def build_items_merge_regression(manifest, dataset, augment_name=None):
    """Build items for MERGE VA regression mode."""
    vmap, amap = dataset.va_maps
    items = []

    for r in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Preparing MERGE VA items",
        leave=False,
    ):
        song_id = str(r.song_id)

        if song_id not in vmap or song_id not in amap:
            continue

        v, a = vmap[song_id], amap[song_id]
        embeddings_path = dataset.embeddings_dir / f"{song_id}.npy"
        if augment_name:
            embeddings_path = dataset.embeddings_dir.parent / f"{dataset.embeddings_dir.name}_{augment_name}" / f"{song_id}_{augment_name}.npy"

        if not embeddings_path.exists():
            continue

        X = np.load(embeddings_path).astype("float32")
        T = len(X)
        Y = np.tile(np.array([v[0], a[0]], dtype="float32"), (T, 1))

        if len(X) == 0:
            continue

        items.append((X, Y))
    return items


def build_items_merge_classification(manifest, dataset, augment_name=None):
    """Build items for MERGE Russell4Q classification mode."""
    qmap = dataset.quadrant_map
    items = []

    for r in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Preparing MERGE Russell4Q items",
        leave=False,
    ):
        song_id = str(r.song_id)

        if song_id not in qmap:
            continue

        q_class = qmap[song_id]
        embeddings_path = dataset.embeddings_dir / f"{song_id}.npy"
        if augment_name:
            embeddings_path = dataset.embeddings_dir.parent / f"{dataset.embeddings_dir.name}_{augment_name}" / f"{song_id}_{augment_name}.npy"

        if not embeddings_path.exists():
            continue

        X = np.load(embeddings_path).astype("float32")
        T = len(X)
        Y = np.full((T,), q_class, dtype="int64")

        if len(X) == 0:
            continue

        items.append((X, Y))
    return items
