import librosa
import numpy as np


class LibrosaMFCCChroma:
    output_dim = 32  # 20 MFCC + 12 Chroma
    frame_hop_seconds = 1.0

    def __init__(self, n_mfcc: int = 20, hop_seconds: float = 1.0):
        self.n_mfcc = int(n_mfcc)
        self.frame_hop_seconds = float(hop_seconds)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        hop = int(sr * self.frame_hop_seconds)

        # mel scale - sounds like how humans hear frequencies
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=hop)
        # convert to dB for easier volume comparison
        S_db = librosa.power_to_db(S)

        # MFCCs capture the overall sound texture/timbre
        mfcc = librosa.feature.mfcc(S=S_db, sr=sr, n_mfcc=self.n_mfcc)
        # chroma captures musical notes and harmony
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop)

        T = min(mfcc.shape[1], chroma.shape[1])
        # combine texture and harmony info into one feature set
        E = np.vstack([mfcc[:, :T], chroma[:, :T]]).T.astype(np.float32)
        return E
