from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


class MERGEDataset:
    """
    MERGE Dataset for Music Emotion Recognition.
    
    Supports two modes:
    - VA (regression): Continuous valence-arousal values
    - Russell4Q (classification): Russell's 4-quadrant emotion model
    
    The dataset contains:
    - merge_audio_complete_av_values.csv: Song ID, Arousal, Valence (0-1 scale)
    - merge_audio_complete_metadata.csv: Song ID, Quadrant (Q1-Q4), and metadata
    - Q1/, Q2/, Q3/, Q4/: Audio files organized by quadrant
    - tvt_dataframes/: Train/validation/test splits
    
    Russell's 4 Quadrants:
    - Q1: High Arousal, High Valence (Happy/Excited)
    - Q2: High Arousal, Low Valence (Angry/Tense)
    - Q3: Low Arousal, Low Valence (Sad/Depressed)
    - Q4: Low Arousal, High Valence (Calm/Relaxed)
    """

    def __init__(
        self, root_dir: Path, out_embeddings_dir: Path, mode: Literal["VA", "Russell4Q"] = "VA",
    ):
        """
        Initialize MERGE dataset.
        
        Args:
            root_dir: Root directory of MERGE dataset
            out_embeddings_dir: Directory containing extracted embeddings
            mode: "VA" for regression, "Russell4Q" for classification
        """
        self.root_dir = root_dir
        self.embeddings_dir = out_embeddings_dir
        self.mode = mode

        # Load annotations
        self.va_file = root_dir / "merge_audio_complete_av_values.csv"
        self.metadata_file = root_dir / "merge_audio_complete_metadata.csv"

        # Audio directories for each quadrant
        self.audio_dirs = {
            "Q1": root_dir / "Q1",
            "Q2": root_dir / "Q2",
            "Q3": root_dir / "Q3",
            "Q4": root_dir / "Q4",
        }

    @staticmethod
    def song_id_from_path(p: Path) -> str:
        """Extract song ID from path (e.g., 'A001' from 'A001.mp3')"""
        return p.stem

    @staticmethod
    def quadrant_to_class(quadrant: str) -> int:
        """Convert quadrant string (Q1-Q4) to class index (0-3)"""
        mapping = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
        return mapping.get(quadrant, -1)

    @staticmethod
    def class_to_quadrant(class_idx: int) -> str:
        """Convert class index (0-3) to quadrant string (Q1-Q4)"""
        mapping = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"}
        return mapping.get(class_idx, "unknown")

    def _read_va_map(self):
        """
        Read VA values from CSV.
        
        Returns:
            (v_map, a_map): Dictionaries mapping song_id -> valence/arousal values
        """
        df = pd.read_csv(self.va_file)
        v_map = {}
        a_map = {}

        for _, row in df.iterrows():
            song_id = str(row["Song"])
            arousal = float(row["Arousal"])  # 0-1 scale
            valence = float(row["Valence"])  # 0-1 scale

            # Convert from [0, 1] to [-1, 1] for consistency with other datasets
            arousal = arousal * 2.0 - 1.0
            valence = valence * 2.0 - 1.0

            # MERGE has static (song-level) annotations, not dynamic
            # Store as single-element arrays for consistency with dynamic datasets
            a_map[song_id] = np.array([arousal], dtype=np.float32)
            v_map[song_id] = np.array([valence], dtype=np.float32)

        return v_map, a_map

    def _read_quadrant_map(self):
        """
        Read quadrant labels from metadata CSV.
        
        Returns:
            q_map: Dictionary mapping song_id -> quadrant class (0-3)
        """
        df = pd.read_csv(self.metadata_file)
        q_map = {}

        for _, row in df.iterrows():
            song_id = str(row["Song"])
            quadrant = str(row["Quadrant"])
            q_map[song_id] = self.quadrant_to_class(quadrant)

        return q_map

    @property
    def va_maps(self):
        """
        Get valence and arousal maps.
        
        Returns:
            (v_map, a_map): Valence and arousal dictionaries
        """
        if self.mode == "VA":
            return self._read_va_map()
        elif self.mode == "Russell4Q":
            # For classification mode, we still return VA maps for compatibility
            # but the training code will use quadrant_map instead
            return self._read_va_map()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def quadrant_map(self):
        """
        Get quadrant class map (only for Russell4Q mode).
        
        Returns:
            q_map: Dictionary mapping song_id -> quadrant class (0-3)
        """
        return self._read_quadrant_map()

    def load_train_val_test_splits(self, split: Literal["70_15_15", "40_30_30"] = "70_15_15"):
        """
        Load predefined train/validation/test splits.
        
        Args:
            split: "70_15_15" or "40_30_30" split ratio
            
        Returns:
            (train_df, val_df, test_df): DataFrames with song IDs for each split
        """
        split_dir = self.root_dir / "tvt_dataframes" / f"tvt_{split}"

        train_df = pd.read_csv(split_dir / f"tvt_{split}_train_audio_complete.csv")
        val_df = pd.read_csv(split_dir / f"tvt_{split}_validate_audio_complete.csv")
        test_df = pd.read_csv(split_dir / f"tvt_{split}_test_audio_complete.csv")

        # Rename columns to match convention
        train_df = train_df.rename(columns={"Song": "song_id", "Quadrant": "quadrant"})
        val_df = val_df.rename(columns={"Song": "song_id", "Quadrant": "quadrant"})
        test_df = test_df.rename(columns={"Song": "song_id", "Quadrant": "quadrant"})

        return train_df, val_df, test_df
