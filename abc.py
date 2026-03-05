import librosa
import os

folder = "audio_dataset/human"  # change if needed
file = os.listdir(folder)[0]

audio_path = os.path.join(folder, file)
y, sr = librosa.load(audio_path, sr=None)

print("Sampling Rate:", sr)


import librosa
import os

folder = "audio_dataset/human"
file = os.listdir(folder)[0]

audio_path = os.path.join(folder, file)
y, sr = librosa.load(audio_path, sr=None)

duration = librosa.get_duration(y=y, sr=sr)
print("Duration (seconds):", duration)