from pathlib import Path

import cv2
from ultralytics import YOLO


CAMERA_INDEX = 0
WINDOW_NAME = "YOLOv8 Pose Skeleton"
MODEL_NAME = "yolov8n-pose.pt"


def main():
    script_dir = Path(__file__).resolve().parent
    local_model_path = script_dir / MODEL_NAME
    model_path = local_model_path if local_model_path.exists() else MODEL_NAME

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다. CAMERA_INDEX={CAMERA_INDEX}")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("프레임을 읽을 수 없습니다.")
                break

            results = model.predict(frame, stream=True, verbose=False)
            annotated_frame = frame

            for result in results:
                annotated_frame = result.plot(boxes=True, labels=True, conf=True, kpt_line=True)
                person_count = 0 if result.keypoints is None else len(result.keypoints)
                cv2.putText(
                    annotated_frame,
                    f"People: {person_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
