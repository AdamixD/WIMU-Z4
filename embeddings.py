from __future__ import annotations

import librosa
import essentia.standard as es
import numpy as np

from typing import Optional
from logging_utils import setup_logger
from config_paths import ESSENTIA_MUSICNN_EMBED_PB


logger = setup_logger()


def load_audio_mono(path, sr=16000, prefer_librosa: bool = False):
    if prefer_librosa:
        try:
            import librosa
            y, _ = librosa.load(path, sr=sr, mono=True)
            return y
        except Exception:
            pass
    try:
        return es.MonoLoader(filename=str(path), sampleRate=sr, resampleQuality=4)()
    except Exception:
        import librosa
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y


class EssentiaMusiCNN:
    name = "essentia_musicnn"
    output_dim = 200
    frame_hop_seconds = 1.0

    def __init__(self, graph_path: Optional[str] = None):
        import essentia.standard as es
        self.es = es
        self.graph_path = str(graph_path) if graph_path else str(ESSENTIA_MUSICNN_EMBED_PB)
        self._embedder = None

    def _get(self):
        if self._embedder is None:
            self._embedder = self.es.TensorflowPredictMusiCNN(
                graphFilename=self.graph_path,
                output="model/dense/BiasAdd"
            )
            logger.info(f"Loaded MusiCNN graph: {self.graph_path}")
        return self._embedder

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return np.asarray(self._get()(audio), dtype=np.float32)


class LibrosaMFCCChroma:
    name = "librosa_mfcc_chroma"
    output_dim = 32  # 20 MFCC + 12 Chroma
    frame_hop_seconds = 1.0

    def __init__(self, n_mfcc: int = 20, hop_seconds: float = 1.0):
        self.librosa = librosa
        self.n_mfcc = int(n_mfcc)
        self.frame_hop_seconds = float(hop_seconds)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        L = self.librosa
        hop = int(sr * self.frame_hop_seconds)

        S = L.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=hop)
        S_db = L.power_to_db(S)

        mfcc = L.feature.mfcc(S=S_db, sr=sr, n_mfcc=self.n_mfcc)
        chroma = L.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop)

        T = min(mfcc.shape[1], chroma.shape[1])
        E = np.vstack([mfcc[:, :T], chroma[:, :T]]).T.astype(np.float32)
        return E


def make_extractor(name: str):
    name = name.lower().strip()
    if name == "essentia_musicnn":
        return EssentiaMusiCNN()
    elif name == "librosa_mfcc_chroma":
        return LibrosaMFCCChroma()
    else:
        raise ValueError(f"Unknown extractor '{name}'. Use 'essentia_musicnn' or 'librosa_mfcc_chroma'.")
