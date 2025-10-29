from __future__ import annotations

import librosa
import numpy as np

from logging_utils import setup_logger


logger = setup_logger()


def load_audio_mono(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


class LibrosaMFCCChroma:
    name = "librosa_mfcc_chroma"
    output_dim = 32  # 20 MFCC + 12 Chroma
    frame_hop_seconds = 1.0

    def __init__(self, n_mfcc: int = 20, hop_seconds: float = 1.0):
        self.n_mfcc = int(n_mfcc)
        self.frame_hop_seconds = float(hop_seconds)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        hop = int(sr * self.frame_hop_seconds)

        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=hop)
        S_db = librosa.power_to_db(S)

        mfcc = librosa.feature.mfcc(S=S_db, sr=sr, n_mfcc=self.n_mfcc)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop)

        T = min(mfcc.shape[1], chroma.shape[1])
        E = np.vstack([mfcc[:, :T], chroma[:, :T]]).T.astype(np.float32)
        return E
