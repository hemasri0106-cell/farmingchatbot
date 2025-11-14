from flask import Flask, render_template, request, jsonify
import requests
import os
import io

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd

app = Flask(__name__)

# ================== SARVAM CONFIG ==================

# Put your real Sarvam API key here OR set SARVAM_API_KEY in your environment.
# Example (Linux/Mac):
#   export SARVAM_API_KEY="your_real_key_here"
SARVAM_API_KEY = "sk_21ltbqrb_DencKzIlH7lSlXNdGxuhuqbp"
SARVAM_BASE_URL = "https://api.sarvam.ai"


# ================== SARVAM CHAT / STT / TTS HELPERS ==================

def call_sarvam_chat(user_message: str) -> str:
    """
    Call Sarvam chat completion API (sarvam-m).
    Responds in the same language as the user's message.
    """
    if not SARVAM_API_KEY:
        return "(Dev mode) Set SARVAM_API_KEY to get real AI answers.\nYou said: " + user_message

    url = f"{SARVAM_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_API_KEY,
    }

    system_prompt = (
        "You are Farm Vaidya, an AI farming assistant for Indian farmers.\n"
        "1. Detect the language of the user's message.\n"
        "2. ALWAYS reply ONLY in the SAME language as the user.\n"
        "3. If the user writes in English (Latin script), you MUST reply in English.\n"
        "4. Do NOT switch to Hindi or any other language unless the user explicitly asks.\n"
        "5. Do NOT add translations to other languages unless requested.\n"
        "Give clear, practical farming advice."
    )

    payload = {
        "model": "sarvam-m",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.4,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_sarvam_stt(audio_file):
    """
    Call Sarvam Speech-to-Text REST API.
    Returns (transcript, language_code).
    """
    if not SARVAM_API_KEY:
        return "", "en-IN"

    url = f"{SARVAM_BASE_URL}/speech-to-text"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
    }

    file_bytes = audio_file.read()
    files = {
        "file": (audio_file.filename or "audio.webm", file_bytes, "audio/webm")
    }

    data = {
        "model": "saarika:v2.5",
        "language_code": "unknown",  # let API auto-detect
    }

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    if resp.status_code != 200:
        print("Sarvam STT error:", resp.status_code, resp.text)
        raise requests.exceptions.HTTPError(
            f"STT returned {resp.status_code}: {resp.text}",
            response=resp,
        )

    result = resp.json()
    transcript = result.get("transcript", "")
    language_code = result.get("language_code") or "en-IN"
    return transcript, language_code


def call_sarvam_tts(text: str, language_code: str):
    """
    Call Sarvam Text-to-Speech REST API.
    Returns a base64-encoded audio string (or None on failure).
    """
    if not SARVAM_API_KEY:
        return None

    url = f"{SARVAM_BASE_URL}/text-to-speech"
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_API_KEY,
    }

    payload = {
        "text": text,
        "target_language_code": language_code,  # e.g. "hi-IN", "en-IN", "ta-IN"
        "model": "bulbul:v2",
        "speaker": "anushka",  # must be a valid Sarvam speaker name
        "speech_sample_rate": 22050,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        print("Sarvam TTS error:", resp.status_code, resp.text)
        raise requests.exceptions.HTTPError(
            f"TTS returned {resp.status_code}: {resp.text}",
            response=resp,
        )

    result = resp.json()
    audios = result.get("audios") or []
    if not audios:
        return None

    # API returns base64-encoded audio bytes in the list
    return audios[0]


# ================== CROP DISEASE MODEL (PlantVillage, 39 classes) ==================

# Model file candidates (use whichever you downloaded/renamed)
MODEL_PATH_CANDIDATES = [
    os.path.join("models", "plant_disease_model_1_latest.pt"),
    os.path.join("models", "plant_disease_model_1.pt"),
    os.path.join("models", "plant_disease_model.pt"),
]

DISEASE_CSV_PATH = "disease_info.csv"
SUPPLEMENT_CSV_PATH = "supplement_info.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 39  # PlantDisease CNN is trained on 39 classes

crop_model = None
disease_info = None
supplement_info = None


class PlantDiseaseCNN(nn.Module):
    """
    Same CNN architecture used in the Plant-Disease-Detection project.
    """

    def __init__(self, K):
        super(PlantDiseaseCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)  # flatten
        out = self.dense_layers(out)
        return out


# Simple resize + tensor (no normalization, matches original project)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def find_model_path():
    for path in MODEL_PATH_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def load_crop_model():
    """
    Load the CNN model and weights once.
    """
    global crop_model

    if crop_model is not None:
        return crop_model

    model_path = find_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "Could not find plant disease model file. "
            "Expected one of: "
            + ", ".join(MODEL_PATH_CANDIDATES)
        )

    model = PlantDiseaseCNN(NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    crop_model = model
    print(f"✅ Loaded crop disease model from {model_path}")
    return crop_model


def load_metadata():
    """
    Load disease_info.csv and supplement_info.csv once (if available).
    These files come from the Plant-Disease-Detection project.
    """
    global disease_info, supplement_info

    if disease_info is None and os.path.exists(DISEASE_CSV_PATH):
        disease_info = pd.read_csv(DISEASE_CSV_PATH, encoding="cp1252")

    if supplement_info is None and os.path.exists(SUPPLEMENT_CSV_PATH):
        supplement_info = pd.read_csv(SUPPLEMENT_CSV_PATH, encoding="cp1252")


def basic_color_analysis(image_bytes):
    """
    Fallback heuristic if the ML model/metadata is not available.
    Simple 'green ratio' analysis.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_small = img.resize((128, 128))
        arr = np.array(img_small, dtype=np.float32) / 255.0

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        green_mask = (g > r) & (g > b)
        green_ratio = float(green_mask.mean())

        if green_ratio > 0.5:
            condition = "Plant looks mostly healthy (lots of green areas detected)."
        elif green_ratio > 0.2:
            condition = "Plant might be mildly stressed (moderate green coverage)."
        else:
            condition = "Plant may be severely stressed or diseased (very low green coverage)."

        return {
            "crop": "Unknown (heuristic only)",
            "condition": condition,
            "green_score": round(green_ratio, 2),
        }

    except Exception as e:
        print("Heuristic image analysis error:", e)
        return {
            "crop": "Unknown",
            "condition": "Could not analyze image. Please try another one.",
        }


def predict_plant_disease(image_bytes):
    """
    Run the uploaded image through the CNN model + metadata
    to get crop, disease name, description, prevention, etc.
    """
    # Load model + metadata (lazy)
    load_metadata()
    model = load_crop_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)            # [1, 39]
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    idx = int(pred_idx.item())
    confidence = float(conf.item())

    # Default values
    crop_name = "Unknown crop"
    condition = "Unknown condition"
    disease_name = f"class_{idx}"
    description = None
    prevention = None
    image_url = None
    supplement_name = None
    supplement_image_url = None
    supplement_buy_link = None

    # Map index → metadata if CSVs are present
    if disease_info is not None and 0 <= idx < len(disease_info):
        row = disease_info.iloc[idx]
        disease_name = str(row.get("disease_name", f"class_{idx}"))
        description = str(row.get("description", "")).strip() or None
        prevention = str(row.get("Possible Steps", "")).strip() or None
        image_url = str(row.get("image_url", "")).strip() or None

        # PlantVillage naming: "Tomato___Early_blight"
        if "___" in disease_name:
            crop_part, disease_part = disease_name.split("___", 1)
        elif "_" in disease_name:
            crop_part, disease_part = disease_name.split("_", 1)
        else:
            crop_part, disease_part = disease_name, ""

        crop_name = crop_part.replace("_", " ").strip().title()
        if disease_part:
            condition = disease_part.replace("_", " ").strip().title()
        else:
            condition = "Healthy"

    # Supplement info (if available and aligned with disease_info index)
    if supplement_info is not None and 0 <= idx < len(supplement_info):
        srow = supplement_info.iloc[idx]
        supplement_name = str(srow.get("supplement name", "")).strip() or None
        supplement_image_url = str(srow.get("supplement image", "")).strip() or None
        supplement_buy_link = str(srow.get("buy link", "")).strip() or None

    result = {
        "crop": crop_name,
        "condition": condition,
        "disease_name": disease_name,
        "description": description,
        "prevention": prevention,
        "image_url": image_url,
        "supplement_name": supplement_name,
        "supplement_image_url": supplement_image_url,
        "supplement_buy_link": supplement_buy_link,
        "confidence": round(confidence, 3),
    }
    return result


def analyze_crop_image(image_file):
    """
    Wrapper: try full CNN model; fall back to color-based heuristic.
    """
    try:
        image_bytes = image_file.read()
        return predict_plant_disease(image_bytes)
    except Exception as e:
        print("Crop ML model error, using heuristic:", e)
        # If we hit an error, we still want to analyze something:
        try:
            image_file.stream.seek(0)
            image_bytes = image_file.read()
        except Exception:
            # if stream is not seekable, just bail with generic message
            return {
                "crop": "Unknown",
                "condition": "Could not analyze image.",
            }
        return basic_color_analysis(image_bytes)


# ================== FLASK ROUTES ==================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"reply": "Please type a message first."}), 400

        bot_reply = call_sarvam_chat(user_message)
        return jsonify({"reply": bot_reply})

    except requests.exceptions.RequestException as api_err:
        print("SARVAM API Error (chat):", api_err)
        return jsonify({"reply": "Error: Unable to connect to Sarvam API."}), 500

    except Exception as e:
        print("Server Error (chat):", e)
        return jsonify({"reply": "Error: Something went wrong on the server."}), 500


@app.route("/chat_audio", methods=["POST"])
def chat_audio():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided."}), 400

        audio_file = request.files["audio"]

        # 1️⃣ Speech → text
        transcript, language_code = call_sarvam_stt(audio_file)
        if not transcript:
            return jsonify({"error": "Could not transcribe the audio."}), 500

        # 2️⃣ Chat in that language
        bot_reply = call_sarvam_chat(transcript)

        # 3️⃣ Text → speech (same language)
        audio_base64 = call_sarvam_tts(bot_reply, language_code)

        return jsonify({
            "user_transcript": transcript,
            "text_reply": bot_reply,
            "audio_base64": audio_base64,
        })

    except requests.exceptions.HTTPError as api_err:
        resp = getattr(api_err, "response", None)
        if resp is not None:
            print("Sarvam API HTTP error:", resp.status_code, resp.text)
            return jsonify({
                "error": f"Sarvam API error ({resp.status_code}): {resp.text}"
            }), 500
        else:
            print("Sarvam API HTTP error:", str(api_err))
            return jsonify({
                "error": f"Sarvam API error: {str(api_err)}"
            }), 500

    except requests.exceptions.RequestException as api_err:
        print("Sarvam API Request error:", api_err)
        return jsonify({
            "error": "Network error while calling Sarvam API. Check your internet connection."
        }), 500

    except Exception as e:
        print("Server Error (audio):", e)
        return jsonify({"error": "Error: Something went wrong on the server."}), 500


@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    """
    Endpoint for crop image upload and ML-based analysis.
    Expects: multipart/form-data with 'image' field.
    """
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        image_file = request.files["image"]
        result = analyze_crop_image(image_file)
        return jsonify(result)

    except Exception as e:
        print("Image analysis error:", e)
        return jsonify({"error": "Error analyzing image."}), 500


if __name__ == "__main__":
    app.run(debug=True)