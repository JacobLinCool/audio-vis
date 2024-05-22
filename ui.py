import gradio as gr
from gen import FILTER_LOWER_BOUND, FILTER_UPPER_BOUND, analyze_audio

with gr.Blocks() as app:
    gr.Markdown(
        """
        # Audio Feature Visualization
        
        Upload an audio file to visualize its features and optionally apply filters. Each visualization helps in understanding different aspects of the audio signal.
        """
    )

    with gr.Row():
        audio = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio File",
        )

    with gr.Row():
        highpass = gr.Slider(
            FILTER_LOWER_BOUND - 1,
            FILTER_UPPER_BOUND,
            step=1,
            label="High-pass filter cutoff frequency (Hz).",
            info="Frequency above which signals are allowed to pass through.",
            value=FILTER_LOWER_BOUND - 1,
        )

        lowpass = gr.Slider(
            FILTER_LOWER_BOUND - 1,
            FILTER_UPPER_BOUND,
            step=1,
            label="Low-pass filter cutoff frequency (Hz).",
            info="Frequency below which signals are allowed to pass through.",
            value=FILTER_LOWER_BOUND - 1,
        )

    with gr.Row():
        bandpass_low = gr.Slider(
            FILTER_LOWER_BOUND - 1,
            FILTER_UPPER_BOUND,
            step=1,
            label="Band-pass filter low cutoff frequency (Hz).",
            info="Lower frequency bound for band-pass filter.",
            value=FILTER_LOWER_BOUND - 1,
        )

        bandpass_high = gr.Slider(
            FILTER_LOWER_BOUND - 1,
            FILTER_UPPER_BOUND,
            step=1,
            label="Band-pass filter high cutoff frequency (Hz).",
            info="Higher frequency bound for band-pass filter.",
            value=FILTER_LOWER_BOUND - 1,
        )

    btn = gr.Button("Visualize Features", variant="primary")

    with gr.Row():
        waveform = gr.Image(
            label="Waveform: Visual representation of the audio signal over time."
        )

    with gr.Row():
        spectrogram = gr.Image(
            label="Spectrogram: Graphical representation of the spectrum of frequencies in a sound signal as they vary with time."
        )
        mfcc = gr.Image(
            label="MFCC: Mel-frequency cepstral coefficients, representing the short-term power spectrum of a sound."
        )

    with gr.Row():
        rms_energy = gr.Image(
            label="RMS Energy: Root Mean Square energy of the audio signal."
        )
        zero_crossing_rate = gr.Image(
            label="Zero Crossing Rate: Rate at which the signal changes from positive to negative or back."
        )

    with gr.Row():
        spectral_centroid = gr.Image(
            label="Spectral Centroid: Indicates where the center of mass of the spectrum is located."
        )
        spectral_bandwidth = gr.Image(
            label="Spectral Bandwidth: The width of a range of frequencies."
        )

    with gr.Row():
        spectral_rolloff = gr.Image(
            label="Spectral Rolloff: Frequency below which a specified percentage of the total spectral energy lies."
        )
        spectral_contrast = gr.Image(
            label="Spectral Contrast: Difference in amplitude between peaks and valleys in a sound spectrum."
        )

    with gr.Row():
        tempo = gr.Image(label="Tempo: Estimated tempo of the audio signal.")
        tempogram = gr.Image(
            label="Tempogram: Localized autocorrelation of the onset strength envelope."
        )

    btn.click(
        fn=analyze_audio,
        inputs=[audio, highpass, lowpass, bandpass_low, bandpass_high],
        outputs=[
            waveform,
            spectrogram,
            mfcc,
            zero_crossing_rate,
            spectral_centroid,
            spectral_bandwidth,
            rms_energy,
            spectral_contrast,
            spectral_rolloff,
            tempo,
            tempogram,
        ],
    )
