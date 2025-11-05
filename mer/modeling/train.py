from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import typer

from mer.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()


@app.command()
def main(
    dataset_name: Annotated[Literal['DEAM',], typer.Option(case_sensitive=False)] = 'DEAM',
    model_path: Path = MODELS_DIR / "model.pkl",
    head: Annotated[Literal['BiGRU',], typer.Option(case_sensitive=False)] = 'BiGRU',
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 6,
    patience: int = 5,
    kfolds: int = 5,
    seed: int = 2024,
    test_size: float = 0.1,
    loss: Annotated[Literal['ccc', 'mse', 'hybrid'], typer.Option(case_sensitive=False)] = 'ccc',
    hidden_dim: int = 128,
    dropout: float = 0.2,
    labels_scale: Annotated[Literal['19', 'norm'], typer.Option(case_sensitive=False, help='Scale of dynamic labels in DEAM source files')] = 'norm',
):
    report_dir = REPORTS_DIR / f'training_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    embeddings_dir = PROCESSED_DATA_DIR / dataset_name / 'embeddings'
    assert embeddings_dir.is_dir(), 'Embeddings dir not found'

    manifest_path = PROCESSED_DATA_DIR / dataset_name / 'manifest.csv'
    assert manifest_path.is_file(), 'Manifest file not found'
    manifest = pd.read_csv(manifest_path)

    logger.info(f'Manifest shape: {manifest.shape}')
    # removed sorting and unique(), data should already be sanitized after extract_embeddings step
    song_ids = manifest['song_id']
    train_ids, test_ids = train_test_split(song_ids, test_size)
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

    folds = []
    for train, test in kf.split(train_ids):
        pass
    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
