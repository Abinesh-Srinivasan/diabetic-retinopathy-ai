import os
import sys
from functools import lru_cache

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from keras.applications import EfficientNetB3
from keras.layers import Concatenate, Dense, GlobalAveragePooling2D, Input
from keras.models import Model, load_model as keras_load_model
from vit_keras import vit


APP_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))


MODEL_PATH = os.path.join(PROJECT_ROOT, "hybrid_best.h5")
IMAGE_SIZE = 224

STAGES = [
    {
        "index": 0,
        "slug": "no-dr",
        "label": "No DR",
        "short_label": "Healthy",
        "severity": "No diabetic retinopathy detected",
        "description": "No visible diabetic retinopathy signs were identified in the retinal image.",
    },
    {
        "index": 1,
        "slug": "mild-npdr",
        "label": "Mild NPDR",
        "short_label": "Mild",
        "severity": "Mild non-proliferative diabetic retinopathy",
        "description": "Small retinal changes may be present, often limited to early microaneurysms.",
    },
    {
        "index": 2,
        "slug": "moderate-npdr",
        "label": "Moderate NPDR",
        "short_label": "Moderate",
        "severity": "Moderate non-proliferative diabetic retinopathy",
        "description": "More widespread retinal damage is likely, with stronger indicators than mild NPDR.",
    },
    {
        "index": 3,
        "slug": "severe-npdr",
        "label": "Severe NPDR",
        "short_label": "Severe",
        "severity": "Severe non-proliferative diabetic retinopathy",
        "description": "The retina shows advanced vessel damage and a high-risk pre-proliferative stage.",
    },
    {
        "index": 4,
        "slug": "pdr",
        "label": "PDR",
        "short_label": "Proliferative",
        "severity": "Proliferative diabetic retinopathy",
        "description": "Abnormal new blood vessel growth is likely, which is considered the most advanced stage.",
    },
]

TEAM_MEMBERS = ["ABINESH S", "DILJAZ R.S", "SURESH S"]
GUIDE = {
    "name": "Mrs.R.Jayalakshmi",
    "designation": "Asst. professor, Computer Science & Engg.",
    "institution": "Rajiv Gandhi College of Engineering and Technology",
}


def preprocess_uploaded_image(file_bytes, image_size=IMAGE_SIZE):
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported image file.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge((l_channel, a_channel, b_channel))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image = image.astype("float32") / 255.0

    return np.expand_dims(image, axis=0)


def build_hybrid_for_inference(num_classes=len(STAGES)):
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    cnn_base = EfficientNetB3(
        weights=None,
        include_top=False,
        input_tensor=inputs,
    )
    cnn_features = GlobalAveragePooling2D()(cnn_base.output)

    vit_base = vit.vit_b16(
        image_size=IMAGE_SIZE,
        pretrained=False,
        include_top=False,
        pretrained_top=False,
    )
    vit_features = vit_base(inputs)

    fused = Concatenate()([cnn_features, vit_features])
    fused = Dense(512, activation="relu")(fused)
    fused = Dense(256, activation="relu")(fused)
    outputs = Dense(num_classes, activation="softmax")(fused)

    return Model(inputs=inputs, outputs=outputs)


@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    try:
        return keras_load_model(MODEL_PATH, compile=False)
    except Exception:
        model = build_hybrid_for_inference(num_classes=len(STAGES))
        model.load_weights(MODEL_PATH)
        return model


def build_prediction_response(probabilities):
    top_index = int(np.argmax(probabilities))
    top_stage = STAGES[top_index]
    confidence = float(probabilities[top_index])

    probability_rows = []
    for stage, score in zip(STAGES, probabilities):
        probability_rows.append(
            {
                "index": stage["index"],
                "label": stage["label"],
                "short_label": stage["short_label"],
                "slug": stage["slug"],
                "severity": stage["severity"],
                "description": stage["description"],
                "probability": round(float(score), 6),
                "percentage": round(float(score) * 100, 2),
            }
        )

    probability_rows.sort(key=lambda row: row["probability"], reverse=True)

    return {
        "prediction": {
            "index": top_stage["index"],
            "label": top_stage["label"],
            "short_label": top_stage["short_label"],
            "slug": top_stage["slug"],
            "severity": top_stage["severity"],
            "description": top_stage["description"],
            "confidence": round(confidence, 6),
            "confidence_percentage": round(confidence * 100, 2),
            "has_dr": top_stage["index"] > 0,
        },
        "probabilities": probability_rows,
        "stages": STAGES,
    }


app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "index.html",
        stages=STAGES,
        team_members=TEAM_MEMBERS,
        guide=GUIDE,
        model_name="Hybrid CNN-ViT",
        model_accuracy="85%",
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload a retinal fundus image."}), 400

    uploaded_file = request.files["image"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No image file was selected."}), 400

    try:
        image = preprocess_uploaded_image(uploaded_file.read())
        model = load_model()
        probabilities = model.predict(image, verbose=0)[0]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(build_prediction_response(probabilities))


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_path": MODEL_PATH,
            "model_exists": os.path.exists(MODEL_PATH),
            "stages": len(STAGES),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
