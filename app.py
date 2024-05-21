import io
import gradio as gr
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from PIL import Image

FILTER_UPPER_BOUND = 20000
FILTER_LOWER_BOUND = 0


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
    y, sr = librosa.load(file)
    y = apply_filters(
        y, sr, highpass_cutoff, lowpass_cutoff, bandpass_low, bandpass_high
    )

    def plot_waveform(y: np.ndarray, sr: int) -> np.ndarray:
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y, sr=sr)
        plt.title("Waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        return plt_to_numpy(plt)

    def plot_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
        plt.figure(figsize=(14, 5))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        return plt_to_numpy(plt)

    def plot_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(mfccs, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title("MFCC")
        return plt_to_numpy(plt)

    def plot_zcr(y: np.ndarray) -> np.ndarray:
        zcr = librosa.feature.zero_crossing_rate(y=y)
        plt.figure(figsize=(14, 5))
        plt.plot(zcr[0])
        plt.title("Zero Crossing Rate")
        plt.xlabel("Frames")
        plt.ylabel("Rate")
        return plt_to_numpy(plt)

    def plot_spectral_centroid(y: np.ndarray, sr: int) -> np.ndarray:
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        plt.figure(figsize=(14, 5))
        plt.semilogy(t, spectral_centroids, label="Spectral centroid")
        plt.xlabel("Time")
        plt.ylabel("Hz")
        plt.title("Spectral Centroid")
        return plt_to_numpy(plt)

    def plot_spectral_bandwidth(y: np.ndarray, sr: int) -> np.ndarray:
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        frames = range(len(spectral_bandwidth))
        t = librosa.frames_to_time(frames)
        plt.figure(figsize=(14, 5))
        plt.semilogy(t, spectral_bandwidth, label="Spectral bandwidth")
        plt.xlabel("Time")
        plt.ylabel("Hz")
        plt.title("Spectral Bandwidth")
        return plt_to_numpy(plt)

    def plot_rms(y: np.ndarray) -> np.ndarray:
        rms = librosa.feature.rms(y=y)[0]
        plt.figure(figsize=(14, 5))
        plt.plot(rms)
        plt.title("RMS Energy")
        plt.xlabel("Frames")
        plt.ylabel("RMS")
        return plt_to_numpy(plt)

    def plot_spectral_contrast(y: np.ndarray, sr: int) -> np.ndarray:
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(spectral_contrast, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title("Spectral Contrast")
        return plt_to_numpy(plt)

    def plot_spectral_rolloff(y: np.ndarray, sr: int) -> np.ndarray:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        frames = range(len(spectral_rolloff))
        t = librosa.frames_to_time(frames)
        plt.figure(figsize=(14, 5))
        plt.semilogy(t, spectral_rolloff, label="Spectral rolloff")
        plt.xlabel("Time")
        plt.ylabel("Hz")
        plt.title("Spectral Rolloff")
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
    )


space = gr.Interface(
    allow_flagging="never",
    fn=analyze_audio,
    inputs=[
        gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio File",
        ),
        gr.Slider(
            FILTER_LOWER_BOUND - 1,
            FILTER_UPPER_BOUND,
            step=1,
            label="High-pass filter cutoff frequency (Hz).",
            info="Frequency above which signals are allowed to pass through.",
            value=FILTER_LOWER_BOUND - 1,
        ),
        gr.Slider(
            FILTER_LOWER_BOUND - 1,
            FILTER_UPPER_BOUND,
            step=1,
            label="Low-pass filter cutoff frequency (Hz).",
            info="Frequency below which signals are allowed to pass through.",
            value=FILTER_LOWER_BOUND - 1,
        ),
        gr.Slider(
            FILTER_LOWER_BOUND - 1,
            FILTER_UPPER_BOUND,
            step=1,
            label="Band-pass filter low cutoff frequency (Hz).",
            info="Lower frequency bound for band-pass filter.",
            value=FILTER_LOWER_BOUND - 1,
        ),
        gr.Slider(
            FILTER_LOWER_BOUND - 1,
            FILTER_UPPER_BOUND,
            step=1,
            label="Band-pass filter high cutoff frequency (Hz).",
            info="Higher frequency bound for band-pass filter.",
            value=FILTER_LOWER_BOUND - 1,
        ),
    ],
    outputs=[
        gr.Image(label=f"{name}: {desc}")
        for name, desc in [
            ("Waveform", "Visual representation of the audio signal over time."),
            (
                "Spectrogram",
                "Graphical representation of the spectrum of frequencies in a sound signal as they vary with time.",
            ),
            (
                "MFCC",
                "Mel-frequency cepstral coefficients, representing the short-term power spectrum of a sound.",
            ),
            (
                "Zero Crossing Rate",
                "Rate at which the signal changes from positive to negative or back.",
            ),
            (
                "Spectral Centroid",
                "Indicates where the center of mass of the spectrum is located.",
            ),
            ("Spectral Bandwidth", "The width of a range of frequencies."),
            ("RMS Energy", "Root Mean Square energy of the audio signal."),
            (
                "Spectral Contrast",
                "Difference in amplitude between peaks and valleys in a sound spectrum.",
            ),
            (
                "Spectral Rolloff",
                "Frequency below which a specified percentage of the total spectral energy lies.",
            ),
        ]
    ],
    title="Audio Feature Visualization",
    description="Upload an audio file to visualize its features and optionally apply filters. Each visualization helps in understanding different aspects of the audio signal.",
)

space.launch()
