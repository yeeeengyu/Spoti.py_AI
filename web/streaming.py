import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from pymongo import MongoClient
import time
import datetime
import os
from dotenv import load_dotenv

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")
mongodb = MongoClient(MONGODB_URL)
db = mongodb['spotipy']
col = db['sleepy']

model = YOLO("./models/bestM.pt")

CONF_THRES = 0.30
IMGSZ = 640
EAR_THRESH = 0.19
SMOOTH_N = 5
SAVE_INTER_SEC = 0.7
FONT = cv2.FONT_HERSHEY_SIMPLEX
lastsave_ts = 0.0

latest_frame = None

def euclidean(p1, p2) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_ear(pts) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4) + 1e-6)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDXS  = [263, 387, 385, 362, 380, 373]

ear_hist = deque(maxlen=SMOOTH_N)

def push_frame(image_bytes: bytes) -> None:
    global latest_frame
    latest_frame = image_bytes

def _placeholder(text: str = "Waiting for frames...") -> bytes:
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(canvas, text, (20, 180), FONT, 0.8, (0, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", canvas)
    return buf.tobytes() if ok else b""

def generate():
    global lastsave_ts
    while True:
        image_bytes = latest_frame
        if not image_bytes:
            frame_bytes = _placeholder()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            frame_bytes = _placeholder("Invalid frame")
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        h, w = frame.shape[:2]
        yolo_res = model(frame, conf=CONF_THRES, imgsz=IMGSZ)
        frame = yolo_res[0].plot()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            def to_xy(idx):
                return (lm[idx].x * w, lm[idx].y * h)
            right_eye_pts = [to_xy(i) for i in RIGHT_EYE_IDXS]
            left_eye_pts  = [to_xy(i) for i in LEFT_EYE_IDXS]
            ear_r = eye_ear(right_eye_pts)
            ear_l = eye_ear(left_eye_pts)
            ear_value = (ear_r + ear_l) / 2.0
            ear_hist.append(ear_value)
            for (x, y) in right_eye_pts + left_eye_pts:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)

        if len(ear_hist) > 0:
            ear_smooth = sum(ear_hist) / len(ear_hist)
            status = "OPEN" if ear_smooth >= EAR_THRESH else "CLOSED"
            now = time.time()
            if now - lastsave_ts >= SAVE_INTER_SEC:
                doc = {"timestamp": datetime.datetime.utcnow(), "ear_smooth": float(ear_smooth), "status": status}
                try:
                    col.insert_one(doc)
                    lastsave_ts = now
                except Exception as e:
                    print(f"[Mongo] insert failed: {e}")

            cv2.putText(frame, f"EAR: {ear_smooth:.3f}  [{status}]",
                        (10, 30), FONT, 0.8, (0, 255, 0) if status=="OPEN" else (0, 0, 255), 2)

            bar_max = 200
            ear_clamped = max(0.0, min(0.4, ear_smooth))
            bar_len = int((ear_clamped / 0.4) * bar_max)
            cv2.rectangle(frame, (10, 40), (10 + bar_max, 60), (50, 50, 50), 1)
            cv2.rectangle(frame, (10, 40), (10 + bar_len, 60),
                          (0, 255, 0) if status=="OPEN" else (0, 0, 255), -1)
        else:
            cv2.putText(frame, "EAR: -- (no face)", (10, 30), FONT, 0.8, (0, 255, 255), 2)

        ok, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes() if ok else _placeholder("Encode failed")
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
