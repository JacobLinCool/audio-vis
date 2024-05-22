import io
import os
import librosa
import librosa.display
import numpy as np
import matplotlib
from matplotlib.font_manager import fontManager
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from PIL import Image

FILTER_UPPER_BOUND = 20000
FILTER_LOWER_BOUND = 0

# use ./fonts/NotoSansTC-Regular.ttf
fontManager.addfont("fonts/NotoSansTC-Regular.ttf")
matplotlib.rc("font", family="Noto Sans TC")


def butter_filter(data: np.ndarray, cutoff: int, fs: int, btype: str, order=5):
    nyquist = 0.5 * fs
    if btype in ["low", "high"]:
        normal_cutoff = cutoff / nyquist
    else:  # 'band'
        normal_cutoff = [c / nyquist for c in cutoff]
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = lfilter(b, a, data)
    return y


def plt_to_numpy(plt: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return np.array(Image.open(buf))


def apply_filters(
    y: np.ndarray,
    sr: int,
    highpass_cutoff: int,
    lowpass_cutoff: int,
    bandpass_low: int,
    bandpass_high: int,
):
    if highpass_cutoff > FILTER_LOWER_BOUND:
        y = butter_filter(y, highpass_cutoff, sr, "high")
    if lowpass_cutoff > FILTER_LOWER_BOUND and lowpass_cutoff < sr / 2:
        y = butter_filter(y, lowpass_cutoff, sr, "low")
    if bandpass_low > FILTER_LOWER_BOUND and bandpass_high < sr / 2:
        y = butter_filter(y, [bandpass_low, bandpass_high], sr, "band")
    return y


def analyze_audio(
    file: str,
    highpass_cutoff: int,
    lowpass_cutoff: int,
    bandpass_low: int,
    bandpass_high: int,
):
    filename = os.path.basename(file)
    y, sr = librosa.load(file)
    y = apply_filters(
        y, sr, highpass_cutoff, lowpass_cutoff, bandpass_low, bandpass_high
    )

    def plot_waveform(y: np.ndarray, sr: int) -> np.ndarray:
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"Waveform ({filename})")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        return plt_to_numpy(plt)

    def plot_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
        plt.figure(figsize=(14, 5))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectrogram ({filename})")
        return plt_to_numpy(plt)

    def plot_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(mfccs, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title(f"MFCC ({filename})")
        return plt_to_numpy(plt)

    def plot_zcr(y: np.ndarray) -> np.ndarray:
        zcr = librosa.feature.zero_crossing_rate(y=y)
        plt.figure(figsize=(14, 5))
        plt.plot(zcr[0])
        plt.title(f"Zero Crossing Rate ({filename})")
        plt.xlabel("Frames")
        plt.ylabel("Rate")
        return plt_to_numpy(plt)

    def plot_spectral_centroid(y: np.ndarray, sr: int) -> np.ndarray:
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        plt.figure(figsize=(14, 5))
        plt.semilogy(t, spectral_centroids, label="Spectral centroid")
        plt.title(f"Spectral Centroid ({filename})")
        plt.xlabel("Time")
        plt.ylabel("Hz")
        return plt_to_numpy(plt)

    def plot_spectral_bandwidth(y: np.ndarray, sr: int) -> np.ndarray:
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        frames = range(len(spectral_bandwidth))
        t = librosa.frames_to_time(frames)
        plt.figure(figsize=(14, 5))
        plt.semilogy(t, spectral_bandwidth, label="Spectral bandwidth")
        plt.title(f"Spectral Bandwidth ({filename})")
        plt.xlabel("Time")
        plt.ylabel("Hz")
        return plt_to_numpy(plt)

    def plot_rms(y: np.ndarray) -> np.ndarray:
        rms = librosa.feature.rms(y=y)[0]
        plt.figure(figsize=(14, 5))
        plt.plot(rms)
        plt.title(f"RMS Energy ({filename})")
        plt.xlabel("Frames")
        plt.ylabel("RMS")
        return plt_to_numpy(plt)

    def plot_spectral_contrast(y: np.ndarray, sr: int) -> np.ndarray:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(spectral_contrast, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title(f"Spectral Contrast ({filename})")
        return plt_to_numpy(plt)

    def plot_spectral_rolloff(y: np.ndarray, sr: int) -> np.ndarray:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        frames = range(len(spectral_rolloff))
        t = librosa.frames_to_time(frames)
        plt.figure(figsize=(14, 5))
        plt.semilogy(t, spectral_rolloff, label="Spectral rolloff")
        plt.xlabel("Time")
        plt.ylabel("Hz")
        plt.title(f"Spectral Rolloff ({filename})")
        return plt_to_numpy(plt)

    def plot_tempo(onset_env: np.ndarray, sr: int) -> np.ndarray:
        dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        frames = range(len(dtempo))
        t = librosa.frames_to_time(frames, sr=sr)
        plt.figure(figsize=(14, 5))
        plt.plot(t, dtempo, label="Tempo")
        plt.title(f"Tempo ({filename})")
        plt.xlabel("Time")
        plt.ylabel("Tempo")
        return plt_to_numpy(plt)

    def plot_tempogram(onset_env: np.ndarray, sr: int) -> np.ndarray:
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(tempogram, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title(f"Tempogram ({filename})")
        return plt_to_numpy(plt)

    waveform = plot_waveform(y, sr)
    spectrogram = plot_spectrogram(y, sr)
    mfcc = plot_mfcc(y, sr)
    zcr = plot_zcr(y)
    spectral_centroid = plot_spectral_centroid(y, sr)
    spectral_bandwidth = plot_spectral_bandwidth(y, sr)
    rms = plot_rms(y)
    spectral_contrast = plot_spectral_contrast(y, sr)
    spectral_rolloff = plot_spectral_rolloff(y, sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = plot_tempo(onset_env, sr)
    tempogram = plot_tempogram(onset_env, sr)

    return (
        waveform,
        spectrogram,
        mfcc,
        zcr,
        spectral_centroid,
        spectral_bandwidth,
        rms,
        spectral_contrast,
        spectral_rolloff,
        tempo,
        tempogram,
    )
