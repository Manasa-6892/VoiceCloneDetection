# AI Voice Clone Detection System

## 📌 Project Overview

The **AI Voice Clone Detection System** is a deep learning-based application designed to detect whether an audio sample is a **real human voice** or an **AI-generated / cloned voice**.

With the rapid advancement of AI voice synthesis technologies, detecting cloned voices has become crucial for preventing **deepfake audio misuse, fraud, and impersonation attacks**.

This system uses **MFCC feature extraction** and a **Convolutional Neural Network (CNN)** model to classify audio samples.

A **Flask web application** provides an interactive interface where users can upload audio files and instantly get prediction results.

---

## 🚀 Features

* Detect **Human Voice vs AI/Cloned Voice**
* Upload `.wav` audio files for prediction
* Deep learning model trained using **CNN**
* **MFCC audio feature extraction**
* User-friendly **Flask web interface**
* **User and Admin role-based access**
* Detection **history tracking**
* Admin panel to view system logs

---

## 🧠 Technologies Used

| Technology         | Purpose                   |
| ------------------ | ------------------------- |
| Python             | Core programming language |
| TensorFlow / Keras | Deep learning model       |
| Flask              | Web application framework |
| Librosa            | Audio processing          |
| NumPy              | Numerical computations    |
| SQLite             | Detection log database    |
| HTML / CSS         | Frontend interface        |

---

## 📂 Project Structure

```
VoiceCloneDetection
│
├── app
│   ├── app.py
│   ├── static
│   │   └── style.css
│   └── templates
│       ├── login.html
│       ├── signup.html
│       ├── dashboard_user.html
│       ├── dashboard_admin.html
│       ├── upload.html
│       ├── detection_detail.html
│       └── profile.html
│
├── feature_extraction
│   └── extract_mfcc.py
│
├── preprocessing
│   └── preprocess_audio.py
│
├── model
│   ├── train_cnn.py
│   └── cnn_voice_model.h5
│
├── requirements.txt
└── README.md
```

---

## 🔬 System Workflow

```
Audio Input (.wav)
        ↓
Audio Preprocessing
        ↓
MFCC Feature Extraction
        ↓
    CNN Model
        ↓
Prediction
(Human Voice / AI Cloned Voice)
        ↓
Flask Web Interface Display
```

---

## 📊 Model Details

| Component          | Description                                |
| ------------------ | ------------------------------------------ |
| Model Type         | Convolutional Neural Network (CNN)         |
| Feature Extraction | MFCC (Mel-Frequency Cepstral Coefficients) |
| Input Length       | 3 seconds audio                            |
| Sampling Rate      | 16000 Hz                                   |
| Framework          | TensorFlow / Keras                         |

The model analyzes spectral patterns of audio signals to identify **synthetic artifacts present in cloned voices**.

---

## 📈 Model Performance

The trained CNN model achieved approximately:

**Accuracy:** ~89%

Performance depends on:

* audio quality
* background noise
* cloning method used

---

## 🖥️ Web Application Features

### User Dashboard

* Upload audio files
* View prediction results
* Access personal detection history

### Admin Dashboard

* View all detection logs
* Filter detections by date or result
* Monitor suspicious cloned voice detections

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/Manasa-6892/VoiceCloneDetection.git
```

```
cd VoiceCloneDetection
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate:

```
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

```
python app/app.py
```

---

### 5️⃣ Open the Application

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🎯 Use Cases

* Deepfake voice detection
* Fraud prevention
* Voice authentication systems
* Media verification
* Cybersecurity applications

---

## 📸 Screenshots

(Add screenshots of your system here)

Example:

```
screenshots/
   login_page.png
   upload_page.png
   prediction_result.png
   admin_dashboard.png
```

---

## 🔮 Future Improvements

* Real-time microphone voice detection
* Larger training dataset
* Advanced transformer-based audio models
* Live streaming voice detection
* Mobile application integration

---

## 👩‍💻 Author

**Manasa Kura**
Computer Science Engineering Student

GitHub:
https://github.com/Manasa-6892

---

## 📜 License

This project is developed for **academic and research purposes**.
