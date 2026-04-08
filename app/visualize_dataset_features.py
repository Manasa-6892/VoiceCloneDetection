import pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Load saved features
with open("features/X_mfcc.pkl", "rb") as f:
    X = pickle.load(f)

with open("features/y_labels.pkl", "rb") as f:
    y = pickle.load(f)

# Separate human and AI features
human_features = X[y == 0]   # label 0 → human
ai_features = X[y == 1]      # label 1 → AI

# Compute average MFCC for each class
human_avg = np.mean(human_features, axis=0)
ai_avg = np.mean(ai_features, axis=0)

# Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
librosa.display.specshow(human_avg, x_axis='time')
plt.title("Average MFCC - Human Voices")
plt.colorbar()

plt.subplot(1,2,2)
librosa.display.specshow(ai_avg, x_axis='time')
plt.title("Average MFCC - AI Voices")
plt.colorbar()

plt.tight_layout()
plt.show()