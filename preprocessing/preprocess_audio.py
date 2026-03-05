import os
import librosa
import numpy as np
import soundfile as sf

# -------- PARAMETERS --------
INPUT_DIR = "audio_dataset"
OUTPUT_DIR = "processed_audio"
SAMPLE_RATE = 16000      # 16 kHz
DURATION = 3             # seconds
MAX_LENGTH = SAMPLE_RATE * DURATION

# Create output folders
os.makedirs(os.path.join(OUTPUT_DIR, "human"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "ai"), exist_ok=True)


def preprocess_audio(input_path, output_path):
    # Load audio
    audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)

    # Normalize audio
    audio = audio / np.max(np.abs(audio))

    # Fix length (pad or trim)
    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]
    else:
        padding = MAX_LENGTH - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')

    # Save processed audio
    sf.write(output_path, audio, SAMPLE_RATE)


def process_folder(label):
    input_folder = os.path.join(INPUT_DIR, label)
    output_folder = os.path.join(OUTPUT_DIR, label)

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            preprocess_audio(input_path, output_path)

    print(f"Preprocessing completed for: {label}")


# -------- MAIN --------
if __name__ == "__main__":
    process_folder("human")
    process_folder("ai")
    print("All audio files preprocessed successfully.")
