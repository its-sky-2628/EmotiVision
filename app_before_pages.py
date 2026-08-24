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
import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from deepface import DeepFace


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze-face", methods=["POST"])
def analyze_face():
    """
    Receive an image from the webcam and analyze facial expression.

    Expected:
        multipart/form-data
        image=<image file>
    """

    if "image" not in request.files:
        return jsonify({
            "error": "No image provided"
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "error": "Empty image file"
        }), 400

    try:
        # Read uploaded image
        image_bytes = image_file.read()

        # Convert bytes -> numpy array
        image_array = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "error": "Could not decode image"
            }), 400

        # Analyze facial expression
        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True
        )

        # DeepFace can return either a dict or list
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        emotion_scores = result.get("emotion", {})

        dominant_emotion = result.get(
            "dominant_emotion",
            "unknown"
        )

        # Convert numpy values to normal Python floats
        emotions = {
            emotion: round(float(score), 2)
            for emotion, score in emotion_scores.items()
        }

        confidence = emotions.get(
            dominant_emotion,
            0
        )

        return jsonify({
            "success": True,
            "emotion": dominant_emotion,
            "confidence": confidence,
            "emotions": emotions
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
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