from ultralytics import YOLO
import cv2
import threading
from collections import deque
import time
import os
import mediapipe as mp
import numpy as np
def euclidean(p1, p2) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_ear(pts) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4) + 1e-6)
MODEL_PATH = os.getenv("MODEL_PATH", "./models/bestM.pt")
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "0"))   # 0번 웹캠
FRAME_W = int(os.getenv("FRAME_W", "640"))
FRAME_H = int(os.getenv("FRAME_H", "640"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.30"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
EAR_THRESH = float(os.getenv("EAR_THRESH", "0.21"))
SMOOTH_N = int(os.getenv("SMOOTH_N", "5"))
FONT = cv2.FONT_HERSHEY_SIMPLEX

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDXS  = [263, 387, 385, 362, 380, 373]

class FrameProcessor:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.cap = cv2.VideoCapture(DEVICE_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        if not self.cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다. DEVICE_INDEX 확인하세요.")

        self.ear_hist = deque(maxlen=SMOOTH_N)
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.running = False
        self.t = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self):
        self.running = False
        if self.t:
            self.t.join(timeout=2)
        if self.cap:
            self.cap.release()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # YOLO 추론 + 그리기
            yolo_res = self.model(frame, conf=CONF_THRES, imgsz=IMGSZ)
            out = yolo_res[0].plot()

            # MediaPipe FaceMesh + EAR
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
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
                self.ear_hist.append(ear_value)

                # 눈 포인트 시각화
                for (x, y) in right_eye_pts + left_eye_pts:
                    cv2.circle(out, (int(x), int(y)), 2, (0, 255, 255), -1)

            # EAR 바/텍스트
            if len(self.ear_hist) > 0:
                ear_smooth = sum(self.ear_hist) / len(self.ear_hist)
                status = "OPEN" if ear_smooth >= EAR_THRESH else "CLOSED"

                cv2.putText(out, f"EAR: {ear_smooth:.3f}  [{status}]",
                            (10, 30), FONT, 0.8,
                            (0, 255, 0) if status == "OPEN" else (0, 0, 255), 2)

                bar_max = 200
                ear_clamped = max(0.0, min(0.4, ear_smooth))
                bar_len = int((ear_clamped / 0.4) * bar_max)
                cv2.rectangle(out, (10, 40), (10 + bar_max, 60), (50, 50, 50), 1)
                cv2.rectangle(out, (10, 40), (10 + bar_len, 60),
                              (0, 255, 0) if status == "OPEN" else (0, 0, 255), -1)
            else:
                cv2.putText(out, "EAR: -- (no face)",
                            (10, 30), FONT, 0.8, (0, 255, 255), 2)

            # JPEG 인코드
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self.lock:
                    self.latest_jpeg = buf.tobytes()
            else:
                # 인코딩 실패시 잠깐 쉼
                time.sleep(0.005)