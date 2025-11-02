# backend/app.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np

from deepface import DeepFace
from transformers import pipeline

app = FastAPI(title="Face+Text Emotion Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

text_emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

class TextRequest(BaseModel):
    text: str

@app.post("/predict_face")
async def predict_face(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    img_array = np.array(img)
    try:
        result = DeepFace.analyze(img_array, actions=["emotion"], enforce_detection=False)
        dominant = result.get("dominant_emotion", None)
        scores = result.get("emotion", {})
        return {"dominant_emotion": dominant, "scores": scores}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_text")
async def predict_text(req: TextRequest):
    text = req.text
    if not text or text.strip() == "":
        return {"error": "Empty text"}
    outputs = text_emotion(text)
    result = {item["label"]: float(item["score"]) for item in outputs}
    dominant = max(result.items(), key=lambda x: x[1])[0] if result else None
    return {"dominant_emotion": dominant, "scores": result}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
