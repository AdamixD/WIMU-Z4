from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
import pandas as pd
import torch
import typer

from mer.config import DEFAULT_DEVICE
from mer.modeling.embeddings import extract_embeddings
from mer.modeling.utils.metrics import labels_convert

app = typer.Typer()


def load_model(model_path: Path, device: str = DEFAULT_DEVICE):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def analyze_audio(song_embds, model, device: str = DEFAULT_DEVICE):
    with torch.no_grad():
        P = model(torch.from_numpy(song_embds).unsqueeze(0).float().to(device))
        P = P.squeeze(0).cpu().numpy().clip(-1.0, 1.0).astype("float32")

    return pd.DataFrame(
        {
            "valence_norm": P[:, 0],
            "arousal_norm": P[:, 1],
            "valence_19": labels_convert(P[:, 0], src="norm", dst="19"),
            "arousal_19": labels_convert(P[:, 1], src="norm", dst="19"),
        }
    )


@app.command(help="Predict dynamic V/A for an audio file (DEAM models).")
def main(
    audio_path: Annotated[Path, typer.Option()],
    model_path: Annotated[Path, typer.Option()],
    device: Annotated[Literal["cuda", "cpu"], typer.Option(case_sensitive=False)] = DEFAULT_DEVICE,
    verbose: bool = typer.Option(is_flag=True, default=False),
):
    logger.info(f"Loading model from {model_path}...")
    model = load_model(model_path)

    song_embds = extract_embeddings(audio_path)

    logger.info("Performing inference...")
    preds = analyze_audio(song_embds, model, device)
    logger.success("Inference complete.")

    predictions_path = model_path.parent / "va_predictions.csv"
    preds.to_csv(predictions_path, index=False)
    logger.info(f"Saved predictions to {predictions_path}")

    if verbose:
        logger.info(
            f"[MEAN ours norm] V={preds['valence_norm'].mean():.3f} A={preds['arousal_norm'].mean():.3f}"
        )
        logger.info(
            f"[MEAN ours 1..9] V={preds['valence_19'].mean():.3f} A={preds['arousal_19'].mean():.3f}"
        )


if __name__ == "__main__":
    app()
