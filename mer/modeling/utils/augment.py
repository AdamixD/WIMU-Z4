import random

import numpy as np
import torch
import torchaudio.functional as F


def np_to_torch_waveform(y: np.ndarray) -> torch.Tensor:
    """Convert 1D numpy audio to torch tensor (1, n_samples)."""
    return torch.from_numpy(y).unsqueeze(0)

def torch_to_np_waveform(w: torch.Tensor) -> np.ndarray:
    """Convert torch waveform tensor (1, n_samples) back to 1D numpy array."""
    return w.detach().cpu().squeeze(0).numpy()

def shift(audio: np.ndarray, max_frac=0.08) -> np.ndarray:
    w = np_to_torch_waveform(audio)
    n = w.shape[-1]
    shift_amt = int(random.uniform(-max_frac, max_frac) * n)
    if shift_amt != 0:
        w = torch.roll(w, shifts=shift_amt, dims=-1)
    return torch_to_np_waveform(w)

def gain(audio: np.ndarray, min_gain_db=-6.0, max_gain_db=6.0) -> np.ndarray:
    w = np_to_torch_waveform(audio)
    gain_db = random.uniform(min_gain_db, max_gain_db)
    factor = 10 ** (gain_db / 20.0)
    w = w * factor
    return torch_to_np_waveform(w)

def reverb(audio: np.ndarray, sr: int, reverberance=50) -> np.ndarray:
    w = np_to_torch_waveform(audio)
    # make a simple exponential decay impulse response
    ir_length = int(0.03 * sr)  # 30 ms
    ir = torch.exp(-torch.linspace(0, 3, ir_length))
    ir = ir.unsqueeze(0).unsqueeze(0)  # (out_ch, in_ch, n)
    w = torch.nn.functional.conv1d(w.unsqueeze(0), ir, padding=ir_length//2).squeeze(0)
    wet = reverberance / 100.0

    if w.shape[-1] != audio.shape[-1]:
        min_len = min(w.shape[-1], audio.shape[-1])
        w = w[..., :min_len]
        audio_t = np_to_torch_waveform(audio)[..., :min_len]
    else:
        audio_t = np_to_torch_waveform(audio)

    w = (1 - wet) * audio_t + wet * w
    return torch_to_np_waveform(w)

def lowpass(audio: np.ndarray, sr: int, cutoff=3000) -> np.ndarray:
    w = np_to_torch_waveform(audio)
    w = F.lowpass_biquad(
        w,
        sample_rate=sr,
        cutoff_freq=cutoff
    )
    return torch_to_np_waveform(w)

def highpass(audio: np.ndarray, sr: int, cutoff=300) -> np.ndarray:
    w = np_to_torch_waveform(audio)
    w = F.highpass_biquad(
        w,
        sample_rate=sr,
        cutoff_freq=cutoff
    )
    return torch_to_np_waveform(w)

def bandpass(audio: np.ndarray, sr: int, f_min=300, f_max=3000) -> np.ndarray:
    w = np_to_torch_waveform(audio)

    w = F.highpass_biquad(
        w,
        sample_rate=sr,
        cutoff_freq=f_min
    )

    w = F.lowpass_biquad(
        w,
        sample_rate=sr,
        cutoff_freq=f_max
    )
    return torch_to_np_waveform(w)

def pitch_shift(audio: np.ndarray, sr: int, n_steps=None) -> np.ndarray:
    if n_steps is None:
        n_steps = random.uniform(-2.0, 2.0)  # semitones
    w = np_to_torch_waveform(audio)
    w_shifted = F.pitch_shift(
        waveform=w,
        sample_rate=sr,
        n_steps=n_steps
    )
    return torch_to_np_waveform(w_shifted)
