# %%
import json
import cv2
import numpy as np
import yaml
import time
from pathlib import Path
from ultralytics import YOLO

# 칼만 필터 기초 클래스 (단순화 버전)
class SimpleKalman:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def predict(self, coord_x, coord_y):
        measurement = np.array([[np.float32(coord_x)], [np.float32(coord_y)]])

        if not self.initialized:
            self.kf.statePost = np.array(
                [[np.float32(coord_x)],
                 [np.float32(coord_y)],
                 [0],
                 [0]], np.float32
            )
            self.initialized = True

        self.kf.predict()
        corrected = self.kf.correct(measurement)
        return int(corrected[0, 0]), int(corrected[1, 0])

# %%
# 1. YOLO 모델 로드 (초경량 Nano 모델)
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "yolov8n.pt"
model = YOLO(str(MODEL_PATH if MODEL_PATH.exists() else "yolov8n.pt"))

# %%
# TODO: camera_info.yaml 에서 캘리브레이션 파라미터를 로드하고
#       camera_matrix, dist_coeffs 를 numpy 배열로 만드세요.
YAML_PATH = SCRIPT_DIR / "camera_info.yaml"
if not YAML_PATH.exists():
    raise FileNotFoundError(f"캘리브레이션 파일이 없습니다: {YAML_PATH}")

with open(YAML_PATH, "r") as f:
    calib = yaml.safe_load(f)

camera_matrix = np.array(calib["camera_matrix"], np.float64)
dist_coeffs = np.array(calib["dist_coeff"], np.float64)

# %%

# 미션 1 — 탐지된 객체까지의 거리 추정
REAL_HEIGHT  = 1.7           # 사람의 평균 키 (미터)
def estimate_distance(frame, x1, y1, x2, y2):
    pixel_height = max(y2 - y1, 1)
    focal_length = camera_matrix[1, 1]
    distance = (REAL_HEIGHT * focal_length) / pixel_height
    cv2.putText(frame, f"{distance:.2f} m",
                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    return distance

# 미션 2 — 특정 클래스 탐지 시 경고 출력
TARGET_LABEL = "cell phone"  # 경고를 발생시킬 클래스 이름
def check_warning(frame, label):
    if label == TARGET_LABEL:
        cv2.putText(frame, "WARNING: cell phone detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        return True
    return False

# 미션 3 — 탐지 결과를 JSON 파일로 저장
detection_data = []
def save_detection(label, x1, y1, x2, y2, cx, cy):
    detection_data.append({
        "time": time.time(),
        "label": label,
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "center": [int(cx), int(cy)],
    })

def save_detection_log(path="detection_log.json"):
    log_path = SCRIPT_DIR / path
    with open(log_path, "w") as f:
        json.dump(detection_data, f, indent=2, ensure_ascii=False)
    print(f"탐지 로그 저장 완료: {log_path}")

# %%
# 웹캠 연결 (0번: 내장 웹캠)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다. CAMERA_INDEX=0")

# 칼만 필터 추적기 생성
tracker = SimpleKalman()

success, frame = cap.read()
if not success:
    cap.release()
    raise RuntimeError("첫 프레임을 읽을 수 없습니다.")

h, w = frame.shape[:2]
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2
)

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        # 왜곡 보정 적용 (Undistort)
        undistorted_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # YOLOv8 객체 탐지
        results = model(undistorted_frame, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 칼만 필터 추적
                px, py = tracker.predict(cx, cy)

                cls = int(box.cls[0])
                label = model.names[cls]

                # 탐지 박스 (파란색)
                cv2.rectangle(undistorted_frame,
                              (x1, y1), (x2, y2), (255, 0, 0), 2)
                # 칼만 필터 예측 점 (빨간색)
                cv2.circle(undistorted_frame, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(undistorted_frame, f"{label} (Tracked)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

                estimate_distance(undistorted_frame, x1, y1, x2, y2)
                check_warning(undistorted_frame, label)
                save_detection(label, x1, y1, x2, y2, cx, cy)

        cv2.imshow("Robot Vision Pipeline", undistorted_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    save_detection_log()
