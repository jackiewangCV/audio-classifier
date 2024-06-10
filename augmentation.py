import librosa
import random
import numpy as np

def _add_noise(signal: np.ndarray, noises: np.ndarray, noise_factor: float):
    noise = noises[random.randint(0, noises.shape[0] - 1)]
    return signal + noise * noise_factor

def _add_white_noise(signal: np.ndarray, noise_factor: float):
    noise = np.random.normal(0, signal.std(), signal.size)
    return signal + noise * noise_factor

def _time_stretch(signal: np.ndarray, stretch_rate: float):
    stretched = librosa.effects.time_stretch(signal, rate=stretch_rate)
    if len(stretched) >= len(signal):
        return stretched[:len(signal)]
    else:
        pad_size = len(signal) - len(stretched)
        return np.pad(stretched, (pad_size // 2, pad_size // 2 + pad_size % 2))

def create_noise_augmenter(noises: np.ndarray, min_noise_factor: float, max_noise_factor: float):
    return lambda signal: _add_noise(signal, noises, random.uniform(min_noise_factor, max_noise_factor))

def create_white_noise_augmenter(min_noise_factor: float, max_noise_factor: float):
    return lambda signal: _add_white_noise(signal, random.uniform(min_noise_factor, max_noise_factor))

def create_time_stretch_augmenter(min_stretch_rate: float, max_stretch_rate: float):
    return lambda signal: _time_stretch(signal, random.uniform(min_stretch_rate, max_stretch_rate))

def create_pitch_shift_augmenter(sample_rate: int, min_n_semitones: int, max_n_semitones: int):
    return lambda signal: librosa.effects.pitch_shift(signal,
                                                      sr=sample_rate,
                                                      n_steps=random.randint(min_n_semitones, max_n_semitones))

def create_invert_polarity_augmenter():
    return lambda signal: signal * -1

def create_gain_augmenter(min_gain_factor: float, max_gain_factor: float):
    return lambda signal: signal * random.uniform(min_gain_factor, max_gain_factor)

def create_rotate_augmenter(min_n_samples: int, max_n_samples: int):
    return lambda signal: np.roll(signal, random.randint(min_n_samples, max_n_samples))
