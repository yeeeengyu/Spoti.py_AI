import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import mediapipe as mp



model = YOLO("./models/bestM.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

CONF_THRES = 0.30
IMGSZ = 640
EAR_THRESH = 0.2
SMOOTH_N = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX

def euclidean(p1, p2) -> int:
    return np.linalg.norm(np.array(p1) - np.array(p2))
def eye_ear(pts) -> int:
    p1, p2, p3, p4, p5, p6 = pts
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4) + 1e-6)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDXS  = [263, 387, 385, 362, 380, 373]

ear_hist = deque(maxlen=SMOOTH_N)

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임 못읽음")
        break
    h, w = frame.shape[:2]
    yolo_res = model(frame, conf=CONF_THRES, imgsz=IMGSZ)
    frame = yolo_res[0].plot()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear_value = None
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

        cv2.putText(frame, f"EAR: {ear_smooth:.3f}  [{status}]",
                    (10, 30), FONT, 0.8, (0, 255, 0) if status=="OPEN" else (0, 0, 255), 2)

        bar_max = 200
        ear_clamped = max(0.0, min(0.4, ear_smooth))
        bar_len = int((ear_clamped / 0.4) * bar_max)
        cv2.rectangle(frame, (10, 40), (10 + bar_max, 60), (50, 50, 50), 1)
        cv2.rectangle(frame, (10, 40), (10 + bar_len, 60),
                      (0, 255, 0) if status=="OPEN" else (0, 0, 255), -1)

    else:
        cv2.putText(frame, "EAR: -- (no face)",
                    (10, 30), FONT, 0.8, (0, 255, 255), 2)

    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(33) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()