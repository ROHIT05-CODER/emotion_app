// 🎥 Face & 💬 Text Emotion Detection Frontend

const video = document.getElementById("video");
const captureBtn = document.getElementById("captureBtn");
const faceResult = document.getElementById("faceResult");
const textInput = document.getElementById("textInput");
const textResult = document.getElementById("textResult");
const analyzeTextBtn = document.getElementById("analyzeTextBtn");

// ✅ Backend URL (FastAPI)
const BACKEND = "http://127.0.0.1:8000";

// 🎭 Emoji Map for Emotion Display
const EMOJI_MAP = {
  happy: "😄",
  sad: "😢",
  angry: "😠",
  surprise: "😲",
  neutral: "😐",
  fear: "😨",
  disgust: "🤢",
  unknown: "🤔",
};

// 🔹 Start Camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    console.log("✅ Camera started successfully.");
  } catch (err) {
    console.error("❌ Camera error:", err);
    faceResult.innerHTML = "⚠️ Please allow camera access and reload the page.";
  }
}
startCamera();

// 🔹 Capture & Analyze Face Emotion
captureBtn.addEventListener("click", async () => {
  if (!video.srcObject) {
    faceResult.innerHTML = "⚠️ Camera not started yet.";
    return;
  }

  faceResult.innerHTML = "🧠 Analyzing face...";

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 320;
  canvas.height = video.videoHeight || 240;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    console.log("📸 Captured blob:", blob);

    if (!blob) {
      faceResult.innerHTML = "❌ Failed to capture image.";
      return;
    }

    const fd = new FormData();
    fd.append("file", blob, "capture.jpg");

    try {
      const resp = await fetch(`${BACKEND}/predict_face`, {
        method: "POST",
        body: fd,
        mode: "cors",
        headers: { Accept: "application/json" },
      });

      console.log("📥 Raw Response:", resp);

      if (!resp.ok) throw new Error(`Response not ok (${resp.status})`);

      const data = await resp.json();
      console.log("🧠 Face data:", data);

      if (data.error) {
        faceResult.innerHTML = `<b>Error:</b> ${data.error}`;
        return;
      }

      const emotion = data.emotion || "unknown";
      const emoji = EMOJI_MAP[emotion.toLowerCase()] || "🤔";
      const stress = data.stress_level || "N/A";
      const suggestion = data.suggestion || "No suggestion provided.";

      faceResult.innerHTML = `
        <h3>${emoji} Emotion: <b>${emotion}</b></h3>
        <b>Stress Level:</b> ${stress}<br/>
        <b>Suggestion:</b> ${suggestion}
      `;
    } catch (err) {
      console.error("❌ Request failed:", err);
      faceResult.innerHTML = `❌ Request failed.<br/>${err.message}`;
    }
  }, "image/jpeg", 0.9);
});

// 🔹 Analyze Text Emotion
analyzeTextBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) {
    alert("Please type some text first!");
    return;
  }

  textResult.innerHTML = "🔍 Analyzing text emotion...";

  try {
    const resp = await fetch(`${BACKEND}/predict_text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ text }),
      mode: "cors",
    });

    console.log("📥 Raw Text Response:", resp);

    if (!resp.ok) throw new Error(`Response not ok (${resp.status})`);

    const data = await resp.json();
    console.log("💬 Text data:", data);

    if (data.error) {
      textResult.innerHTML = `<b>Error:</b> ${data.error}`;
      return;
    }

    const emotion = data.dominant_emotion || data.emotion || "unknown";
    const emoji = EMOJI_MAP[emotion.toLowerCase()] || "🤔";
    const suggestion = data.suggestion || "No suggestion available.";

    textResult.innerHTML = `
      <h3>${emoji} Dominant Emotion: <b>${emotion}</b></h3>
      <b>Suggestion:</b> ${suggestion}<br/><br/>
      <b>Scores:</b>
    `;

    const div = document.createElement("div");
    if (data.scores) {
      Object.entries(data.scores).forEach(([k, v]) => {
        const p = document.createElement("div");
        p.textContent = `${k}: ${(v * 100).toFixed(1)}%`;
        div.appendChild(p);
      });
    } else {
      div.textContent = "No score data available.";
    }

    textResult.appendChild(div);
  } catch (err) {
    console.error("❌ Text request error:", err);
    textResult.innerHTML = `⚠️ Error contacting backend.<br/>${err.message}`;
  }
});
