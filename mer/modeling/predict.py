import json
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
import typer

from mer.heads import BiGRUHead
from mer.modeling.embeddings import extract_embeddings
from mer.modeling.utils.metrics import labels_convert

app = typer.Typer()


@app.command(help='Predict dynamic V/A for an audio file (DEAM models).')
def main(
    audio_path: Path,
    training_dir: Path,
    device: Annotated[
        Literal['cuda', 'cpu'], typer.Option(case_sensitive=False)] = (
    'cuda' if torch.cuda.is_available() else 'cpu'),
    verbose: bool = typer.Option(is_flag=True, default=False),
):
    checkpkts = json.loads((training_dir/'best_checkpoints.json').read_text())

    song_embds = extract_embeddings(audio_path)
    in_dim = song_embds.shape[1]

    models = []
    for p in tqdm(checkpkts, leave=False):
        sd = torch.load(p, map_location=device)['model']
        model = BiGRUHead(in_dim=in_dim).to(device)
        model.load_state_dict(sd)
        model.eval()
        models.append(model)

    logger.info('Performing inference for model...')
    with torch.no_grad():
        preds = [
            m(torch.from_numpy(song_embds).unsqueeze(0).float().to(device)).squeeze(
                0).cpu().numpy() for m in models]
        P = np.mean(preds, axis=0) if len(preds) > 1 else preds[0]

    P = np.clip(P, -1.0, 1.0).astype("float32")
    logger.success('Inference complete.')

    out_data = {
        'valence_norm': P[:, 0],
        'arousal_norm': P[:, 1],
        'valence_19': labels_convert(P[:, 0], src='norm', dst='19'),
        'arousal_19': labels_convert(P[:, 1], src='norm', dst='19'),
    }

    predictions_path = training_dir/'va_predictions.csv'

    df = pd.DataFrame(out_data)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(predictions_path, index=False)
    logger.info(f'Saved predictions to {predictions_path}')

    if verbose:
        logger.info(
            f"[MEAN ours norm] V={df['valence_norm'].mean():.3f} A={df['arousal_norm'].mean():.3f}")
        logger.info(
            f"[MEAN ours 1..9] V={df['valence_19'].mean():.3f} A={df['arousal_19'].mean():.3f}")


if __name__ == '__main__':
    app()
