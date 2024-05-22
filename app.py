import gradio as gr
from gen import FILTER_LOWER_BOUND, FILTER_UPPER_BOUND, analyze_audio

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

space.queue(status_update_rate=10.0, max_size=10).launch()
