import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# -------- LOAD FEATURES --------
with open("features/X_mfcc.pkl", "rb") as f:
    X = pickle.load(f)

with open("features/y_labels.pkl", "rb") as f:
    y = pickle.load(f)

print("Original X shape:", X.shape)

# -------- PREPARE DATA --------
# Add channel dimension for CNN
X = X[..., np.newaxis]  # (samples, 40, 130, 1)

# One-hot encode labels
y = to_categorical(y, num_classes=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------- BUILD CNN MODEL --------
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(40, 130, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------- TRAIN MODEL --------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -------- EVALUATE MODEL --------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# -------- SAVE MODEL --------
model.save("model/cnn_voice_model.h5")
print("Model saved as cnn_voice_model.h5")
