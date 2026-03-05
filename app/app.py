import os
from datetime import datetime, timezone, timedelta

import numpy as np
import librosa
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    current_user,
    logout_user,
    UserMixin,
)
from flask_bcrypt import Bcrypt
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


app = Flask(__name__)

# ----------------- CONFIG -----------------

app.config["SECRET_KEY"] = "change-this-secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///voice_clone_detection.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

TEST_UPLOAD_FOLDER = os.path.join("uploads", "test_audios")
os.makedirs(TEST_UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = TEST_UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"wav"}

# Try both app/model and project_root/model so it works regardless of where
# the model directory is placed relative to this file.
_model_candidates = [
    os.path.join(BASE_DIR, "model", "cnn_voice_model.h5"),
    os.path.join(PROJECT_ROOT, "model", "cnn_voice_model.h5"),
]
MODEL_PATH = next((p for p in _model_candidates if os.path.exists(p)), _model_candidates[0])
SAMPLE_RATE = 16000
DURATION = 3
MAX_LENGTH = SAMPLE_RATE * DURATION
N_MFCC = 40
MAX_FRAMES = 130

IST = timezone(timedelta(hours=5, minutes=30))

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

model = load_model(MODEL_PATH)


# ----------------- MODELS -----------------


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), default="user")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    result_label = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    detection_type = db.Column(db.String(20), nullable=False)
    is_suspicious = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("detections", lazy=True))

    # Convenience aliases matching the updated specification
    @property
    def audio_name(self):
        return self.file_name

    @property
    def prediction_label(self):
        return self.result_label


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    detection_id = db.Column(db.Integer, db.ForeignKey("detection_log.id"), nullable=True)

    user = db.relationship("User", backref=db.backref("notifications", lazy=True))
    detection = db.relationship("DetectionLog", backref=db.backref("notifications", lazy=True))


# ----------------- TEMPLATE HELPERS -----------------


def utc_to_ist(dt):
    if not dt:
        return None
    # All timestamps are stored in UTC; convert to IST for display.
    return dt.replace(tzinfo=timezone.utc).astimezone(IST)


@app.template_filter("ist_date")
def ist_date(dt):
    local = utc_to_ist(dt)
    return local.strftime("%Y-%m-%d") if local else ""


@app.template_filter("ist_time")
def ist_time(dt):
    local = utc_to_ist(dt)
    return local.strftime("%H:%M:%S") if local else ""


@app.context_processor
def inject_navbar_data():
    if current_user.is_authenticated:
        notifications = (
            Notification.query.filter_by(user_id=current_user.id)
            .order_by(Notification.created_at.desc())
            .limit(5)
            .all()
        )
        unread_exists = any(not n.is_read for n in notifications)
    else:
        notifications = []
        unread_exists = False

    return dict(
        navbar_notifications=notifications,
        navbar_unread_exists=unread_exists,
    )


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ----------------- AUDIO / MODEL -----------------


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]
    else:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)), mode="constant")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] > MAX_FRAMES:
        mfcc = mfcc[:, :MAX_FRAMES]
    else:
        mfcc = np.pad(
            mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])), mode="constant"
        )

    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc


def predict_file(file_path):
    features = preprocess_audio(file_path)
    prediction = model.predict(features)[0]

    idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    if idx == 0:
        label = "Human Voice"
    else:
        label = "Cloned (AI) Voice"

    return label, confidence


# ----------------- AUTH -----------------


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        requested_role = (request.form.get("role") or "").lower()

        if not username or not password:
            flash("Username and password are required.")
            return redirect(url_for("signup"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists.")
            return redirect(url_for("signup"))

        # Only allow an "admin" signup if there is no admin yet.
        role = "user"
        if requested_role == "admin":
            existing_admin = User.query.filter_by(role="admin").first()
            if not existing_admin:
                role = "admin"

        pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        user = User(username=username, password_hash=pw_hash, role=role)
        db.session.add(user)
        db.session.commit()
        flash("Account created. Please log in.")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/notifications/<int:notif_id>/read")
@login_required
def mark_notification_read(notif_id: int):
    notif = Notification.query.get_or_404(notif_id)
    if notif.user_id != current_user.id and current_user.role != "admin":
        flash("You cannot modify this notification.")
        return redirect(url_for("dashboard"))

    notif.is_read = True
    db.session.commit()

    next_url = request.args.get("next") or url_for("dashboard")
    return redirect(next_url)


@app.route("/notifications/<int:notif_id>/open")
@login_required
def open_notification(notif_id: int):
    notif = Notification.query.get_or_404(notif_id)
    if notif.user_id != current_user.id and current_user.role != "admin":
        flash("You cannot view this notification.")
        return redirect(url_for("dashboard"))

    notif.is_read = True
    db.session.commit()

    if notif.detection_id:
        return redirect(url_for("detection_detail", detection_id=notif.detection_id))

    return redirect(url_for("dashboard"))


# ----------------- PAGES -----------------


@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    if current_user.role == "admin":
        return render_template("dashboard_admin.html")
    return render_template("dashboard_user.html")


# ----------------- DETECTION -----------------


@app.route("/upload")
@login_required
def upload():
    return render_template("upload.html")


@app.route("/detect", methods=["POST"])
@login_required
def detect():
    if "audio" not in request.files:
        flash("No audio file provided.")
        return redirect(url_for("dashboard"))

    file = request.files["audio"]
    if file.filename == "":
        flash("No selected file.")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Only .wav allowed.")
        return redirect(url_for("dashboard"))

    original_name = secure_filename(file.filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_name = f"{current_user.id}_{timestamp}_{original_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    label, confidence = predict_file(save_path)
    is_suspicious = label == "Cloned (AI) Voice" and confidence >= 0.8

    log = DetectionLog(
        user_id=current_user.id,
        file_name=unique_name,
        file_path=save_path,
        result_label=label,
        confidence=confidence,
        detection_type="upload",
        is_suspicious=is_suspicious,
    )
    db.session.add(log)
    # Create notification for cloned voices
    if label == "Cloned (AI) Voice":
        notif_msg = f"⚠ Cloned voice detected in file: {original_name}"
        notif = Notification(
            user_id=current_user.id,
            message=notif_msg,
            is_read=False,
            detection_id=log.id,
        )
        db.session.add(notif)
    db.session.commit()

    return redirect(url_for("detection_detail", detection_id=log.id))


@app.route("/test_audios/<filename>")
@login_required
def serve_audio(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/detection/<int:detection_id>")
@login_required
def detection_detail(detection_id):
    log = DetectionLog.query.get_or_404(detection_id)

    if current_user.role != "admin" and log.user_id != current_user.id:
        flash("You cannot view this detection.")
        return redirect(url_for("dashboard"))

    return render_template("detection_detail.html", log=log)


@app.route("/history")
@login_required
def history_user():
    logs = (
        DetectionLog.query.filter_by(user_id=current_user.id)
        .order_by(DetectionLog.created_at.desc())
        .all()
    )
    return render_template("detection_history_user.html", logs=logs)


@app.route("/admin/history")
@login_required
def history_admin():
    if current_user.role != "admin":
        flash("Admin access only.")
        return redirect(url_for("dashboard"))

    query = DetectionLog.query

    date_str = request.args.get("date")
    detection_type = request.args.get("detection_type")
    suspicious_only = request.args.get("suspicious") == "1"

    if date_str:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            start = datetime.combine(date_obj, datetime.min.time())
            end = datetime.combine(date_obj, datetime.max.time())
            query = query.filter(
                DetectionLog.created_at >= start, DetectionLog.created_at <= end
            )
        except ValueError:
            pass

    if detection_type:
        query = query.filter_by(detection_type=detection_type)

    if suspicious_only:
        query = query.filter_by(is_suspicious=True)

    logs = query.order_by(DetectionLog.created_at.desc()).all()
    return render_template("detection_history_admin.html", logs=logs)


@app.route("/profile")
@login_required
def profile():
    logs = (
        DetectionLog.query.filter_by(user_id=current_user.id)
        .order_by(DetectionLog.created_at.desc())
        .all()
    )
    return render_template("profile.html", logs=logs)


@app.route("/admin/details")
@login_required
def admin_details():
    if current_user.role != "admin":
        flash("Admin access only.")
        return redirect(url_for("dashboard"))

    query = DetectionLog.query.join(User, DetectionLog.user_id == User.id)

    search = request.args.get("search", "").strip()
    prediction = request.args.get("prediction")
    date_str = request.args.get("date")

    if search:
        like_pattern = f"%{search}%"
        query = query.filter(DetectionLog.file_name.ilike(like_pattern))

    if prediction == "Human":
        query = query.filter(DetectionLog.result_label == "Human Voice")
    elif prediction == "Cloned":
        query = query.filter(DetectionLog.result_label == "Cloned (AI) Voice")

    if date_str:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            start = datetime.combine(date_obj, datetime.min.time())
            end = datetime.combine(date_obj, datetime.max.time())
            query = query.filter(
                DetectionLog.created_at >= start, DetectionLog.created_at <= end
            )
        except ValueError:
            pass

    logs = query.order_by(DetectionLog.created_at.desc()).all()
    return render_template("admin_details.html", logs=logs)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
