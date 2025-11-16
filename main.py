import os
import base64
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load HuggingFace API Key
HF_API_KEY = os.getenv("HF_API_KEY")

# Model URL – using GFPGAN for face restoration
HF_API_URL = "https://api-inference.huggingface.co/models/TencentARC/GFPGAN"

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ▼ Allow all origins (localhost, file://, etc)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "ReviveMyPhoto API Running"}


@app.post("/api/enhance")
async def enhance_image(file: UploadFile = File(...)):
    # Read the uploaded image as bytes
    image_bytes = await file.read()

    # Request to Hugging Face Model
    response = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        files={"file": ("input.jpg", image_bytes, "image/jpeg")},
    )

    # Error Handling
    if response.status_code != 200:
        return JSONResponse(
            {
                "error": "Model processing failed",
                "details": response.text,
                "status": response.status_code,
            },
            status_code=500,
        )

    # Convert raw output image into Base64
    restored_image_b64 = base64.b64encode(response.content).decode("utf-8")

    return {
        "restored_image": f"data:image/jpeg;base64,{restored_image_b64}"
    }
