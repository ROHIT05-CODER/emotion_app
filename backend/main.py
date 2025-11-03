from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from transformers import pipeline
import cv2
import numpy as np
import uvicorn
import traceback
import os

# 🎯 Create FastAPI app
app = FastAPI(
    title="Emotion Detection API 😎",
    description="Detects facial emotions and stress levels, and gives helpful suggestions 💬",
    version="1.0.4",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json"
)

# ✅ Allow all origins (for testing or frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Load the text emotion model
print("🚀 Loading emotion model (this may take a few seconds)...")
text_emotion = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)
print("✅ Text emotion model loaded successfully!")


@app.get("/")
def root():
    return {"message": "😎 Emotion & Stress Detection backend is running successfully!"}


# 🧩 FACE EMOTION DETECTION
@app.post("/predict_face")
async def predict_face(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image file. Please upload a valid face image."}

        result = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)

        if not result or not isinstance(result, list) or 'dominant_emotion' not in result[0]:
            return {"error": "No face detected. Try with better lighting or clearer image."}

        emotion = result[0]['dominant_emotion']

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
            "emotion": str(emotion),
            "stress_level": str(stress_level),
            "suggestion": str(suggestion)
        }

    except Exception as e:
        print("❌ Face analysis failed:", str(e))
        traceback.print_exc()
        return {"success": False, "error": f"Face analysis failed: {str(e)}"}


# ✍️ TEXT EMOTION DETECTION
@app.post("/predict_text")
async def predict_text(data: dict):
    text = data.get("text", "").strip()
    if not text:
        return {"success": False, "error": "Please enter some text."}

    try:
        result = text_emotion(text)
    except Exception as e:
        print("❌ Text analysis failed:", str(e))
        traceback.print_exc()
        return {"success": False, "error": f"Text analysis failed: {str(e)}"}

    # Flatten nested result
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        result = result[0]

    scores = {}
    for item in result:
        if isinstance(item, dict) and "label" in item and "score" in item:
            scores[str(item["label"])] = float(item["score"])

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
        "dominant_emotion": str(dominant_emotion),
        "scores": {k: float(v) for k, v in scores.items()},
        "suggestion": str(suggestion)
    }


# 🚀 RUN APP (Render-compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # ✅ Render will auto-assign this
    uvicorn.run("main:app", host="0.0.0.0", port=port)
