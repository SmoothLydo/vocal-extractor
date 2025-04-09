# infer_vocals.py

import sys
import os

# Append the root directory of your project (which contains model/, utils/, etc.)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import VocalSeparatorCNN  # make sure your model definition is here
from utils.pre_processing import load_audio, audio_to_mel, normalize_spectrograms, get_song_title  # import your actual functions
from utils.reconstruct_audio import unnormalize_spectrograms, save_spec_plot

import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy.io.wavfile as wav

# Suppress deprication warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*librosa.core.audio.__audioread_load.*")

# Gets the path the model .pth file from the project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # goes one level up from /main
PATH_TO_MODEL_SAVE = os.path.join(ROOT_DIR, "model", "vocal_separator_model_V18.pth")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")


# Example terminal command for windows:
# python vocal_sep.py "C:/vocal_sep_spec/songs/Clara Berry And Wooldog - Air Traffic.stem.mp4"


def main():
    if len(sys.argv) != 2:
        print("Usage: python infer_vocals.py <path_to_audio>")
        sys.exit(1)

    audio_path = sys.argv[1]
    song_title = get_song_title(audio_path)

    print(f"Extracting vocals from \"{song_title}\"")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (device.type == "cuda"):
        print("CUDA enabled\n")
    else:
        print("CUDA not enabled\n")

    # Load model
    model = VocalSeparatorCNN().to(device)
    model.load_state_dict(torch.load(PATH_TO_MODEL_SAVE, map_location=device))
    model.eval()

    # Prepare a list to hold the audio chunks
    all_spec_chunks = []

    # Get total duration of the song
    audio, sr = librosa.load(audio_path, sr=44100, mono=False)
    total_duration = librosa.get_duration(y=audio, sr=sr)
    chunk_duration = 10.0

    # Calculate number of chunks needed
    chunks = int(np.ceil(total_duration / chunk_duration))

    # Process all 10 second chunks of the song
    for i in range(chunks):
        # Load 10 seconds of audio, starting at the appropriate time
        start_time = i * 10  # start at 0s, 10s, 20s for each chunk
        audio = load_audio(audio_path, duration=10.0, offset=start_time)
        
        # Convert audio to mel spectrogram
        mel_spec = audio_to_mel(audio)
        
        # Normalize the spectrogram
        norm_spec, left_minmax, right_minmax = normalize_spectrograms(mel_spec)

        # Convert to tensor
        input_tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, 2, 256, time)

        # Inference
        with torch.no_grad():
            predicted_vocals_chunk = model(input_tensor)
            predicted_vocals_chunk_np = predicted_vocals_chunk[0].detach().cpu().numpy().astype(np.float32)

        # Append the reconstructed audio to the list of chunks
        all_spec_chunks.append(predicted_vocals_chunk_np)
        print(f"Chunk {i+1}/{chunks} Completed")

    full_vocal_spec = np.concatenate(all_spec_chunks, axis=2)  # shape: (2, 256, time_frames)

    # Get the left channel spectrogram (shape: (256, time_frames))
    left_spec = full_vocal_spec[0]  # First channel is the left channel

    # Compress frequency bins from 256 to 16
    compressed_left_spec = left_spec.reshape(16, 16, -1).mean(axis=1)  # shape: (16, time_frames)

    # Flatten the spectrogram for saving
    flattened_left_spec = compressed_left_spec.reshape(16, -1)  # shape: (16, time_frames)

    # Save raw binary data
    raw_path = os.path.join(OUTPUT_DIR, f"{song_title}_left_channel_spectrogram.raw")
    flattened_left_spec.astype(np.float32).tofile(raw_path)

    # Save the shape of the spectrogram
    shape_path = os.path.join(OUTPUT_DIR, f"{song_title}_left_channel_spectrogram_shape.txt")
    with open(shape_path, "w") as f:
        f.write(f"({flattened_left_spec.shape[0]},{flattened_left_spec.shape[1]})")

    # Save the plot (optionally, plot the compressed version instead)
    #save_spec_plot(OUTPUT_DIR, song_title, compressed_left_spec)  # Change left_spec to compressed_left_spec if you want to plot the compressed version

    print(f"Left channel spectrogram raw data and shape saved to {OUTPUT_DIR}")





if __name__ == "__main__":
    main()