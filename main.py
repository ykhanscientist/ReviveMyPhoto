from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import requests
from io import BytesIO

app = FastAPI()

# Allow your frontend (GitHub Pages / Render) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face model API
HF_API_URL = "https://api-inference.huggingface.co/models/TencentARC/GFPGAN"
HF_API_KEY = "YOUR_HUGGINGFACE_API_KEY"  # <--- Replace this!

headers = {"Authorization": f"Bearer {HF_API_KEY}"}


@app.get("/")
async def read_root():
    return {"message": "ReviveMyPhoto API Running"}


@app.post("/api/enhance")
async def enhance(photo: UploadFile = File(...)):
    try:
        img_bytes = await photo.read()

        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=img_bytes
        )

        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={"error": "AI Processing Failed", "details": response.text}
            )

        result_bytes = response.content
        return StreamingResponse(BytesIO(result_bytes), media_type="image/png")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
