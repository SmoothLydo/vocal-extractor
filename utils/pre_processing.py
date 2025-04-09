import librosa
import librosa.display
import numpy as np
import os


def load_audio(filepath, duration=10.0, sr=44100, offset=0.0):
    audio, file_sr = librosa.load(filepath, sr=sr, mono=False, duration=duration, offset=offset)

    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)

    # Pad if the audio is shorter than expected (i.e., the final chunk)
    expected_len = int(sr * duration)
    current_len = audio.shape[1]

    if current_len < expected_len:
        pad_len = expected_len - current_len
        audio = np.pad(audio, ((0, 0), (0, pad_len)), mode='constant')

    return audio.T  # shape: (samples, 2)


def get_song_title(audio_path):
    # Extract filename from path
    filename = os.path.basename(audio_path)

    # Remove the .stem.mp4 extension
    if filename.endswith(".stem.mp4"):
        title = filename[:-9]  # removes last 10 characters
    elif (filename.endswith(".mp4") or filename.endswith(".wav")):
        title = filename[:-4]  # removes last 10 characters
    else:
        title = filename

    return title


def audio_to_mel(audio, sr=44100, n_fft=2048, hop_length=512, n_mels=256):
    # Separate channels
    left_channel = librosa.feature.melspectrogram(y=audio[:, 0], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    right_channel = librosa.feature.melspectrogram(y=audio[:, 1], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert to log scale
    max_value = np.max(left_channel)
    #print(max_value)

    log_left = librosa.power_to_db(left_channel, ref=np.max)
    log_right = librosa.power_to_db(right_channel, ref=np.max)

    # Create stereo log mel spectrogram
    log_mel_spec = np.stack([log_left, log_right], axis=0)
    return log_mel_spec


def normalize_spectrograms(spec, epsilon=1e-6):
    # If spec is a NumPy array, don't use .cpu().numpy()
    left_spec = spec[0]
    right_spec = spec[1]

    # Normalize left channel
    left_min = np.min(left_spec)
    left_max = np.max(left_spec)
    left_range = left_max - left_min
    norm_left = (left_spec - left_min) / (left_range + epsilon)

    # Normalize right channel
    right_min = np.min(right_spec)
    right_max = np.max(right_spec)
    right_range = right_max - right_min
    norm_right = (right_spec - right_min) / (right_range + epsilon)

    # Return normalized spectrograms as a NumPy array
    norm_spec = np.stack([norm_left, norm_right], axis=0)

    # Return normalized spectrograms and their min/max for later denormalization
    return norm_spec, (left_min, left_max), (right_min, right_max)