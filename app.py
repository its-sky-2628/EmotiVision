"""
app.py
------
Flask web app + REST API for facial expression / emotion detection.

Run:
    python app.py

Open:
    http://127.0.0.1:5001
"""

import os
import sqlite3
import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from deepface import DeepFace
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "emotivision-development-secret")
CORS(app)

def init_auth_db():
    conn = sqlite3.connect("users.db")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.commit()
    conn.close()

init_auth_db()



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/features")
def features_page():
    return render_template("features.html")

@app.route("/how-it-works")
def how_it_works_page():
    return render_template("how_it_works.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/blog")
def blog_page():
    return render_template("blog.html")

@app.route("/auth")
def auth_page():
    return render_template("auth.html")

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not name or not email or not password:
        return jsonify({
            "success": False,
            "error": "Name, email and password are required."
        }), 400

    if len(password) < 6:
        return jsonify({
            "success": False,
            "error": "Password must contain at least 6 characters."
        }), 400

    try:
        conn = sqlite3.connect("users.db")
        conn.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, generate_password_hash(password))
        )
        conn.commit()
        conn.close()

        session["user_name"] = name
        session["user_email"] = email

        return jsonify({
            "success": True,
            "message": "Account created successfully.",
            "name": name
        })

    except sqlite3.IntegrityError:
        return jsonify({
            "success": False,
            "error": "This email is already registered."
        }), 409

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:
        return jsonify({
            "success": False,
            "error": "Email and password are required."
        }), 400

    conn = sqlite3.connect("users.db")
    row = conn.execute(
        "SELECT name, email, password FROM users WHERE email = ?",
        (email,)
    ).fetchone()
    conn.close()

    if not row or not check_password_hash(row[2], password):
        return jsonify({
            "success": False,
            "error": "Invalid email or password."
        }), 401

    session["user_name"] = row[0]
    session["user_email"] = row[1]

    return jsonify({
        "success": True,
        "message": "Login successful.",
        "name": row[0]
    })


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({
        "success": True
    })


@app.route("/api/me")
def current_user():
    if "user_email" not in session:
        return jsonify({
            "logged_in": False
        })

    return jsonify({
        "logged_in": True,
        "name": session.get("user_name"),
        "email": session.get("user_email")
    })

@app.route("/api/analyze-face", methods=["POST"])
def analyze_face():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image provided"
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty image file"
        }), 400

    try:
        image_bytes = image_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Could not decode image"
            }), 400

        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False
        )

        if isinstance(results, dict):
            results = [results]

        faces = []

        for number, result in enumerate(results, start=1):
            region = result.get("region") or {}
            emotions_raw = result.get("emotion") or {}

            width = int(region.get("w", 0) or 0)
            height = int(region.get("h", 0) or 0)

            if width <= 0 or height <= 0:
                continue

            emotions = {}

            for name, score in emotions_raw.items():
                emotions[str(name)] = round(float(score), 2)

            dominant = str(
                result.get("dominant_emotion", "neutral")
            ).lower()

            confidence = round(
                float(emotions.get(dominant, 0)),
                2
            )

            faces.append({
                "id": number,
                "emotion": dominant,
                "confidence": confidence,
                "emotions": emotions,
                "region": {
                    "x": int(region.get("x", 0) or 0),
                    "y": int(region.get("y", 0) or 0),
                    "w": width,
                    "h": height
                }
            })

        if not faces:
            return jsonify({
                "success": False,
                "face_count": 0,
                "faces": [],
                "error": "No face detected"
            })

        primary = max(
            faces,
            key=lambda face: face["confidence"]
        )

        return jsonify({
            "success": True,
            "emotion": primary["emotion"],
            "confidence": primary["confidence"],
            "emotions": primary["emotions"],
            "face_count": len(faces),
            "faces": faces
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Face Emotion Detection API"
    })




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))

    print("\n======================================")
    print("   FACE EMOTION DETECTION AI")
    print("======================================")
    print(f"Server running at:")
    print(f"http://127.0.0.1:{port}")
    print("======================================\n")

    app.run(
        debug=True,
        host="0.0.0.0",
        port=port
    )