from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger

from mer.config import DEFAULT_DEVICE
from mer.modeling.embeddings import extract_embeddings
from mer.modeling.utils.metrics import (
    labels_convert,
    quadrant_to_name,
    va_to_russell_quadrant,
)

app = typer.Typer()


def load_model(model_path: Path, device: str = DEFAULT_DEVICE):
    model = torch.load(model_path, map_location=device)

    # Fix for models saved before adding self.act attribute
    if hasattr(model, "out") and not hasattr(model, "act"):
        import torch.nn as nn

        # Determine if it's a classification or regression model
        if model.out.out_features == 4:
            # Classification model - no activation needed (raw logits)
            model.act = nn.Identity()
        else:
            # Regression model - use Tanh
            model.act = nn.Tanh()
        logger.info("Added missing activation layer to loaded model")

    model.eval()
    return model


def analyze_audio_classification(
    song_embds, model, device: str = DEFAULT_DEVICE, return_average: bool = False
):
    """
    Analyze audio and predict Russell 4Q classes.
    
    Args:
        song_embds: Audio embeddings (T, feat_dim)
        model: Trained classification model
        device: Device to run inference on
        return_average: If True, also return track-level statistics
    
    Returns:
        DataFrame with Russell 4Q predictions
    """
    with torch.no_grad():
        logits = model(torch.from_numpy(song_embds).unsqueeze(0).float().to(device))
        logits = logits.squeeze(0).cpu().numpy()

    predicted_classes = np.argmax(logits, axis=1)
    predicted_names = [quadrant_to_name(int(c)) for c in predicted_classes]

    result = {
        "russell4q_class": predicted_classes,
        "russell4q_name": predicted_names,
    }

    df = pd.DataFrame(result)

    if return_average:
        avg_data = {}

        # Dominant class (most frequent)
        class_counts = df["russell4q_class"].value_counts()
        dominant_class = int(class_counts.index[0])
        avg_data["dominant_russell4q_class"] = dominant_class
        avg_data["dominant_russell4q_name"] = quadrant_to_name(dominant_class)
        avg_data["dominant_russell4q_percentage"] = float(class_counts.iloc[0] / len(df) * 100)

        # Class distribution
        for class_idx in range(4):
            class_name = quadrant_to_name(class_idx)
            count = int(class_counts.get(class_idx, 0))
            percentage = float(count / len(df) * 100)
            avg_data[f"{class_name}_count"] = count
            avg_data[f"{class_name}_percentage"] = percentage

        df.attrs["averages"] = avg_data

        logger.info("Track-level statistics:")
        logger.info(
            f"  Dominant quadrant: {avg_data['dominant_russell4q_name']} ({avg_data['dominant_russell4q_percentage']:.1f}%)"
        )
        for class_idx in range(4):
            class_name = quadrant_to_name(class_idx)
            logger.info(f"  {class_name}: {avg_data[f'{class_name}_percentage']:.1f}%")

    return df


def analyze_audio(
    song_embds,
    model,
    device: str = DEFAULT_DEVICE,
    return_average: bool = False,
    map_to_russell4q: bool = False,
):
    """
    Analyze audio and predict VA values.
    
    Args:
        song_embds: Audio embeddings (T, feat_dim)
        model: Trained model
        device: Device to run inference on
        return_average: If True, also return track-level average predictions
        map_to_russell4q: If True, map VA predictions to Russell 4Q quadrants
    
    Returns:
        DataFrame with predictions (and optionally averages and/or Russell 4Q mappings)
    """
    with torch.no_grad():
        P = model(torch.from_numpy(song_embds).unsqueeze(0).float().to(device))
        P = P.squeeze(0).cpu().numpy().clip(-1.0, 1.0).astype("float32")

    result = {
        "valence_norm": P[:, 0],
        "arousal_norm": P[:, 1],
        "valence_19": labels_convert(P[:, 0], src="norm", dst="19"),
        "arousal_19": labels_convert(P[:, 1], src="norm", dst="19"),
    }

    # Add Russell 4Q mapping if requested
    if map_to_russell4q:
        quadrants = va_to_russell_quadrant(P[:, 0], P[:, 1])
        result["russell4q_class"] = quadrants
        result["russell4q_name"] = quadrant_to_name(quadrants)

    df = pd.DataFrame(result)

    # Add average predictions if requested
    if return_average:
        avg_data = {
            "average_valence_norm": df["valence_norm"].mean(),
            "average_arousal_norm": df["arousal_norm"].mean(),
            "average_valence_19": df["valence_19"].mean(),
            "average_arousal_19": df["arousal_19"].mean(),
        }

        if map_to_russell4q:
            # Determine dominant quadrant for the track
            quadrant_counts = df["russell4q_class"].value_counts()
            dominant_quadrant = quadrant_counts.idxmax()
            avg_data["dominant_russell4q_class"] = dominant_quadrant
            avg_data["dominant_russell4q_name"] = quadrant_to_name(dominant_quadrant)

            # Also add quadrant based on average VA
            avg_quadrant = va_to_russell_quadrant(
                avg_data["average_valence_norm"], avg_data["average_arousal_norm"]
            )
            avg_data["average_based_russell4q_class"] = int(avg_quadrant)
            avg_data["average_based_russell4q_name"] = quadrant_to_name(avg_quadrant)

        # Store averages as attributes
        df.attrs["averages"] = avg_data

        logger.info("Track-level averages:")
        for key, value in avg_data.items():
            if isinstance(value, (int, np.integer)):
                logger.info(f"  {key}: {value}")
            elif isinstance(value, str):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value:.4f}")

    return df


@app.command(help="Predict emotion from audio file (VA or Russell4Q, auto-detected).")
def main(
    audio_path: Annotated[Path, typer.Option(help="Path to audio file")],
    model_path: Annotated[Path, typer.Option(help="Path to trained model (.pth)")],
    output_path: Annotated[Path, typer.Option(help="Output CSV path")] = None,
    device: Annotated[Literal["cuda", "cpu"], typer.Option(case_sensitive=False)] = DEFAULT_DEVICE,
    return_average: bool = typer.Option(
        False, "--return-average", help="Return track-level statistics"
    ),
    map_to_russell4q: bool = typer.Option(
        False, "--map-to-russell4q", help="Map VA predictions to Russell 4Q (VA models only)"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Print detailed information"),
):
    """
    Predict emotion for an audio file. Model type (VA/Russell4Q) is automatically detected.
    
    For VA models (2 outputs):
      - Returns valence and arousal values
      - Optionally maps to Russell 4Q quadrants with --map-to-russell4q
    
    For Russell4Q models (4 outputs):
      - Returns quadrant classifications directly
      - --map-to-russell4q is ignored
    
    Examples:
        # Basic prediction (auto-detects model type)
        python -m mer.modeling.predict --audio-path song.mp3 --model-path model.pth
        
        # With track-level statistics
        python -m mer.modeling.predict --audio-path song.mp3 --model-path model.pth --return-average
        
        # VA model with Russell 4Q mapping
        python -m mer.modeling.predict --audio-path song.mp3 --model-path va_model.pth --map-to-russell4q
        
        # Detailed output
        python -m mer.modeling.predict --audio-path song.mp3 --model-path model.pth --return-average --verbose
    """
    logger.info(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Detect model type based on output size
    out_features = model.out.out_features
    if out_features == 2:
        model_type = "VA"
        logger.info("Detected model type: VA (regression)")
    elif out_features == 4:
        model_type = "Russell4Q"
        logger.info("Detected model type: Russell4Q (classification)")
    else:
        raise ValueError(
            f"Unexpected output size: {out_features}. Expected 2 (VA) or 4 (Russell4Q)"
        )

    logger.info(f"Extracting embeddings from {audio_path}...")
    song_embds = extract_embeddings(audio_path)

    logger.info("Performing inference...")
    if model_type == "Russell4Q":
        # Native Russell4Q classification
        if map_to_russell4q:
            logger.warning(
                "--map-to-russell4q is ignored for Russell4Q models (already classification)"
            )
        preds = analyze_audio_classification(
            song_embds, model, device, return_average=return_average
        )
    else:
        # VA regression (optionally map to Russell4Q)
        preds = analyze_audio(
            song_embds,
            model,
            device,
            return_average=return_average,
            map_to_russell4q=map_to_russell4q,
        )
    logger.success("Inference complete.")

    # Determine output path
    if output_path is None:
        if model_type == "Russell4Q":
            output_path = model_path.parent / "russell4q_predictions.csv"
        elif map_to_russell4q:
            output_path = model_path.parent / "va_russell4q_predictions.csv"
        else:
            output_path = model_path.parent / "va_predictions.csv"

    # Save predictions
    preds.to_csv(output_path, index=False)
    logger.info(f"Saved frame-by-frame predictions to {output_path}")

    # Save averages if computed
    if return_average and hasattr(preds, "attrs") and "averages" in preds.attrs:
        avg_output_path = output_path.parent / f"{output_path.stem}_averages.json"
        import json

        with open(avg_output_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            avg_data = {}
            for k, v in preds.attrs["averages"].items():
                if isinstance(v, (np.integer, np.floating)):
                    avg_data[k] = float(v)
                else:
                    avg_data[k] = v
            json.dump(avg_data, f, indent=2)
        logger.info(f"Saved track-level averages to {avg_output_path}")

    if verbose:
        logger.info("\n" + "=" * 50)
        logger.info("FRAME-BY-FRAME STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Number of frames: {len(preds)}")

        if model_type == "Russell4Q":
            # Russell4Q classification statistics
            logger.info("\nRUSSELL 4Q DISTRIBUTION")
            logger.info("=" * 50)
            quadrant_counts = preds["russell4q_name"].value_counts()
            for q_name, count in quadrant_counts.items():
                percentage = (count / len(preds)) * 100
                logger.info(f"{q_name}: {count} frames ({percentage:.1f}%)")
        else:
            # VA regression statistics
            logger.info(
                f"[MEAN norm] V={preds['valence_norm'].mean():.3f} A={preds['arousal_norm'].mean():.3f}"
            )
            logger.info(
                f"[MEAN 1-9]  V={preds['valence_19'].mean():.2f} A={preds['arousal_19'].mean():.2f}"
            )
            logger.info(
                f"[STD norm]  V={preds['valence_norm'].std():.3f} A={preds['arousal_norm'].std():.3f}"
            )

            if map_to_russell4q:
                logger.info("\nRUSSELL 4Q DISTRIBUTION")
                logger.info("=" * 50)
                quadrant_counts = preds["russell4q_name"].value_counts()
                for q_name, count in quadrant_counts.items():
                    percentage = (count / len(preds)) * 100
                    logger.info(f"{q_name}: {count} frames ({percentage:.1f}%)")

        logger.info("=" * 50)


if __name__ == "__main__":
    app()
