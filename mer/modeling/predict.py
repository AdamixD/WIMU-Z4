from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
import pandas as pd
import torch
import typer

from mer.modeling.embeddings import extract_embeddings
from mer.modeling.utils.metrics import labels_convert

app = typer.Typer()


@app.command(help="Predict dynamic V/A for an audio file (DEAM models).")
def main(
    audio_path: Annotated[Path, typer.Option()] = "/mnt/c/WIMU-Z4/data/raw/DEAM/audio/2.mp3",
    model_path: Annotated[Path, typer.Option()] = "/mnt/c/WIMU-Z4/reports/training_2025-11-09_17-51-45",
    device: Annotated[Literal["cuda", "cpu"], typer.Option(case_sensitive=False)] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
    verbose: bool = typer.Option(is_flag=True, default=False),
):
    logger.info(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location=device)
    model.eval()

    song_embds = extract_embeddings(audio_path)

    logger.info("Performing inference...")
    with torch.no_grad():
        P = model(torch.from_numpy(song_embds).unsqueeze(0).float().to(device))
        P = P.squeeze(0).cpu().numpy().clip(-1.0, 1.0).astype("float32")
    logger.success("Inference complete.")

    out_data = {
        "valence_norm": P[:, 0],
        "arousal_norm": P[:, 1],
        "valence_19": labels_convert(P[:, 0], src="norm", dst="19"),
        "arousal_19": labels_convert(P[:, 1], src="norm", dst="19"),
    }

    predictions_path = model_path.parent / "va_predictions.csv"
    df = pd.DataFrame(out_data)
    df.to_csv(predictions_path, index=False)
    logger.info(f"Saved predictions to {predictions_path}")

    if verbose:
        logger.info(
            f"[MEAN ours norm] V={df['valence_norm'].mean():.3f} A={df['arousal_norm'].mean():.3f}"
        )
        logger.info(
            f"[MEAN ours 1..9] V={df['valence_19'].mean():.3f} A={df['arousal_19'].mean():.3f}"
        )


if __name__ == "__main__":
    app()
