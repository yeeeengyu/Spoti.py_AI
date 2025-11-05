from fastapi import FastAPI, UploadFile, File, Request, Header, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pymongo import MongoClient
from dotenv import load_dotenv
from streaming import push_frame, generate
import httpx
import os

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")
NEST_BASE_URL = os.getenv("NEST_BASE_URL", "https://your-nest.example.com")

client = MongoClient(MONGODB_URL)
db = client['spotipy']
col = db['sleepy']

app = FastAPI()

def _require_bearer(authorization: str | None) -> str:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    return authorization  # 'Bearer xxx' 문자열 그대로 반환

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
    authorization: str | None = Header(None),
):
    _ = _require_bearer(authorization)
    image_bytes = await (file.read() if file else request.body())
    if not image_bytes:
        return JSONResponse({"ok": False, "error": "empty body"}, status_code=400)
    push_frame(image_bytes)
    return {"ok": True}

@app.post("/statistic")
async def statistic(
    authorization: str | None = Header(None),
    limit: int = Query(50, ge=1, le=500),
):
    # 1) 토큰 확인
    auth = _require_bearer(authorization)

    # 2) DB 조회
    docs = list(col.find().sort("timestamp", -1).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])

    # 3) Nest로 포워딩 (토큰은 Authorization 헤더, 데이터는 JSON 바디)
    url = f"{NEST_BASE_URL}/statistic"
    payload = {"data": docs}  # Nest에서 기대하는 키 이름에 맞춰주세요
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                url,
                headers={"Authorization": auth},
                json=payload,
            )
        # Nest 응답 그대로 전달(필요 시 가공 가능)
        content = r.json() if r.headers.get("content-type", "").startswith("application/json") and r.content else {"ok": r.is_success}
        return JSONResponse(status_code=r.status_code, content=content)
    except httpx.RequestError as e:
        # 네트워크/타임아웃 등 오류 처리
        return JSONResponse(status_code=502, content={"ok": False, "error": f"nest_unreachable: {str(e)}"})
