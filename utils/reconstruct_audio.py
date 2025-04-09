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



def save_spec_plot(output_dir, song_title, left_spec, sr=44100, hop_length=512):
    # Separate left and right channels

    # Create fake mel bin frequencies (since you averaged them)
    # This lets us still show y_axis='mel' without an error
    mel_bins = 32
    mel_frequencies = librosa.mel_frequencies(n_mels=mel_bins, fmin=0, fmax=sr // 2)

    # Plot and save the mel spectrogram images
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(left_spec, sr=sr, hop_length=hop_length, x_axis='time',
                              y_axis=None, cmap='magma', fmin=0, fmax=sr // 2)
    plt.yticks(ticks=np.linspace(0, 31, 6), labels=[f"{int(f)}Hz" for f in np.linspace(0, sr//2, 6)])
    plt.title("Predicted Vocal Log Mel Spectrogram (Left Channel, 32 bins)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{song_title}_vocal_spec_left.png"))
    plt.close()
