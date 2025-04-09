import librosa
import numpy as np
import librosa.display
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt

def unnormalize_spectrograms(norm_spec, left_min, left_max, right_min, right_max):
    # Denormalize left channel
    left_spec = norm_spec[0] * (left_max - left_min) + left_min

    # Denormalize right channel
    right_spec = norm_spec[1] * (right_max - right_min) + right_min

    # Return denormalized spectrogram
    return np.stack([left_spec, right_spec], axis=0)



def save_spec_plot(output_dir, song_title, full_vocal_spec):
    # Separate left and right channels
    left_spec = full_vocal_spec[0]
    right_spec = full_vocal_spec[1]

    # Plot and save the mel spectrogram images
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(left_spec, sr=44100, hop_length=512, y_axis='mel', x_axis='time')
    plt.title("Predicted Vocal Log Mel Spectrogram (Left Channel)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{song_title}_vocal_spec_left.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(right_spec, sr=44100, hop_length=512, y_axis='mel', x_axis='time')
    plt.title("Predicted Vocal Log Mel Spectrogram (Right Channel)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{song_title}_vocal_spec_right.png"))
    plt.close()