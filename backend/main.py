from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from transformers import pipeline
import cv2
import numpy as np
import uvicorn
import traceback
import os
import threading
import logging
import torch

# =========================================
# 🌟 FastAPI App Setup
# =========================================
app = FastAPI(
    title="Emotion Detection API 😎",
    description="Detects facial emotions and text emotions, gives stress suggestions 💬",
    version="1.1.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json"
)

# =========================================
# 🌍 CORS Middleware
# =========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now; can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# 🧠 Global Variables
# =========================================
text_emotion = None
face_model_ready = False

# Set cache path for Render or server restarts
os.environ["TRANSFORMERS_CACHE"] = "/opt/render/project/.cache/huggingface"
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

# =========================================
# 🧩 Logging Config
# =========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================
# 🔄 Background Model Preloading
# =========================================
def preload_models():
    global text_emotion, face_model_ready
    try:
        logger.info("🚀 Preloading DeepFace and Transformer models in background...")

        # Load DeepFace model
        DeepFace.build_model("Emotion")
        face_model_ready = True
        logger.info("✅ DeepFace face emotion model loaded successfully!")

        # Load HuggingFace model
        device = 0 if torch.cuda.is_available() else -1
        text_emotion = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=device
        )
        logger.info("✅ Text emotion model loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Model preload failed: {e}")
        traceback.print_exc()


# =========================================
# 🏠 Root Endpoint
# =========================================
@app.get("/")
def root():
    return {"message": "😎 Emotion & Stress Detection backend is running successfully!"}


# =========================================
# 💡 Status Endpoint
# =========================================
@app.get("/status")
def status():
    return {
        "text_model_loaded": text_emotion is not None,
        "face_model_loaded": face_model_ready
    }


# =========================================
# 🧩 FACE EMOTION DETECTION
# =========================================
@app.post("/predict_face")
async def predict_face(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image file. Please upload a valid face image."}

        result = DeepFace.analyze(
            img_path=img, actions=['emotion'], enforce_detection=False
        )

        if not result or not isinstance(result, list) or 'dominant_emotion' not in result[0]:
            return {"error": "No face detected. Try with better lighting or clearer image."}

        emotion = result[0]['dominant_emotion'].lower()

        # Stress logic
        if emotion in ["angry", "fear", "sad"]:
            stress_level = "High"
            suggestion = "You seem stressed. Try deep breathing or calming music 🎵"
        elif emotion == "neutral":
            stress_level = "Medium"
            suggestion = "You seem calm. Stay mindful and take short breaks ☕"
        elif emotion in ["happy", "surprise"]:
            stress_level = "Low"
            suggestion = "You look great! Keep smiling and spread positivity 😊"
        else:
            stress_level = "Unknown"
            suggestion = "Unable to determine emotion clearly. Try retaking the photo."

        return {
            "success": True,
            "emotion": emotion,
            "stress_level": stress_level,
            "suggestion": suggestion
        }

    except Exception as e:
        logger.error(f"❌ Face analysis failed: {e}")
        traceback.print_exc()
        return {"success": False, "error": f"Face analysis failed: {str(e)}"}


# =========================================
# ✍️ TEXT EMOTION DETECTION
# =========================================
@app.post("/predict_text")
async def predict_text(data: dict):
    global text_emotion

    if text_emotion is None:
        return {"success": False, "error": "Model is still loading. Try again in a few seconds."}

    text = data.get("text", "").strip()
    if not text:
        return {"success": False, "error": "Please enter some text."}

    try:
        result = text_emotion(text)

        # Flatten nested result
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            result = result[0]

        scores = {}
        for item in result:
            if isinstance(item, dict) and "label" in item and "score" in item:
                scores[item["label"]] = float(item["score"])

        dominant_emotion = max(scores, key=scores.get) if scores else "unknown"

        # Suggestion logic
        if dominant_emotion in ["anger", "fear", "sadness"]:
            suggestion = "You seem stressed. Try deep breathing or calming music 🎵"
        elif dominant_emotion in ["neutral", "surprise"]:
            suggestion = "You seem calm. Stay mindful and take short breaks ☕"
        elif dominant_emotion in ["joy", "love"]:
            suggestion = "You sound happy! Keep spreading positive energy 😊"
        else:
            suggestion = "Couldn't interpret emotion clearly. Try a longer message."

        return {
            "success": True,
            "dominant_emotion": dominant_emotion,
            "scores": scores,
            "suggestion": suggestion
        }

    except Exception as e:
        logger.error(f"❌ Text analysis failed: {e}")
        traceback.print_exc()
        return {"success": False, "error": f"Text analysis failed: {str(e)}"}


# =========================================
# 🚀 Main (Render-compatible)
# =========================================
if __name__ == "__main__":
    logger.info("✅ FastAPI starting — loading models in background...")
    threading.Thread(target=preload_models, daemon=True).start()
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
