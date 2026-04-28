import cv2       # OpenCV 라이브러리 임포트
import numpy as np  # 행렬 연산을 위한 NumPy

def edge_detection_pipeline():
    # 1. 웹캠 연결 (0번은 내장 웹캠)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cv2.namedWindow('Controls')
    cv2.createTrackbar('Low Threshold',  'Controls', 100, 255, lambda x: None)
    cv2.createTrackbar('High Threshold', 'Controls', 200, 255, lambda x: None)

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # BGR → HSV 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 회색 범위 정의 (낮은 채도 = 무채색)
        lower_gray = np.array([0,   0,  50])
        upper_gray = np.array([180, 40, 200])

        # 마스크 생성 및 적용
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        masked = cv2.bitwise_and(gray, gray, mask=mask)

        blurred = cv2.GaussianBlur(masked, (5, 5), 20)

        low  = cv2.getTrackbarPos('Low Threshold',  'Controls')
        high = cv2.getTrackbarPos('High Threshold', 'Controls')
        # 마스킹 된 영역에만 에지 검출 적용
        edges = cv2.Canny(blurred, low, high)

        # 5. 결과 표시
        cv2.imshow('Original Video', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('masked', masked)
        cv2.imshow('Edge Detection Pipeline', edges)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):  # 's' 키: 현재 프레임 저장
            cv2.imwrite('result.png', edges)
            print("에지 이미지 저장 완료: result.png")
        if key == ord('q'):  # 'q' 키: 종료
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    edge_detection_pipeline()