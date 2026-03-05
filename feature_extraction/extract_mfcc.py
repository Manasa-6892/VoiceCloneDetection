import os
import librosa
import numpy as np
import pickle

# -------- PARAMETERS --------
INPUT_DIR = "processed_audio"
OUTPUT_DIR = "features"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 130   # fixed time dimension

os.makedirs(OUTPUT_DIR, exist_ok=True)

X = []
y = []

label_map = {
    "human": 0,
    "ai": 1
}


def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    # Fix MFCC length (pad or trim)
    if mfcc.shape[1] > MAX_FRAMES:
        mfcc = mfcc[:, :MAX_FRAMES]
    else:
        pad_width = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

    return mfcc


def process_folder(label):
    folder_path = os.path.join(INPUT_DIR, label)
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            mfcc = extract_mfcc(file_path)
            X.append(mfcc)
            y.append(label_map[label])


# -------- MAIN --------
if __name__ == "__main__":
    process_folder("human")
    process_folder("ai")

    X_np = np.array(X)
    y_np = np.array(y)

    # Save features
    with open(os.path.join(OUTPUT_DIR, "X_mfcc.pkl"), "wb") as f:
        pickle.dump(X_np, f)

    with open(os.path.join(OUTPUT_DIR, "y_labels.pkl"), "wb") as f:
        pickle.dump(y_np, f)

    print("MFCC feature extraction completed.")
    print("Feature shape:", X_np.shape)
    print("Labels shape:", y_np.shape)
