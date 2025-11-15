from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Header
from pymongo import MongoClient
from dotenv import load_dotenv
from streaming import push_frame, generate
import httpx
import os

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")
NEST_BASE_URL = os.getenv("NEST_BASE_URL", "https://your-nest.example.com")
security = HTTPBearer(bearerFormat="JWT")

client = MongoClient(MONGODB_URL)
db = client["spotipy"]
col = db["sleepy"]

app = FastAPI()

@app.get("/")
async def root():
    return "ok"

@app.get("/stream")
async def stream():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile | None = File(None),
    credentials: HTTPAuthorizationCredentials = Depends(security), 
):
    auth = f"Bearer {credentials.credentials}"
    image_bytes = await (file.read() if file else request.body())
    if not image_bytes:
        return JSONResponse({"ok": False, "error": "empty body"}, status_code=400)
    push_frame(image_bytes)
    return {"ok": True}

@app.post("/statistic")
async def statistic(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    auth = f"Bearer {credentials.credentials}"

    docs = list(col.find().sort("timestamp", -1).limit(21))
    for d in docs:
        d["_id"] = str(d["_id"])

    url = f"{NEST_BASE_URL}/statistic"
    payload = {"data": docs}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(url, headers={"Authorization": auth}, json=payload)
        content = (
            r.json()
            if r.headers.get("content-type", "").startswith("application/json") and r.content
            else {"ok": r.is_success}
        )
        return JSONResponse(status_code=r.status_code, content=content)
    except httpx.RequestError as e:
        return JSONResponse(status_code=502, content={"ok": False, "error": f"nest_unreachable: {str(e)}"})

@app.get("/latest-status")
async def latest_status(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    # 필요하면 토큰 꺼내서 쓰면 됨
    # token = credentials.credentials

    doc = col.find_one(sort=[("timestamp", -1)])

    if not doc:
        return {"ok": True, "status": None}

    return {
        "ok": True,
        "status": doc.get("status"),
        "timestamp": doc.get("timestamp")
    }
