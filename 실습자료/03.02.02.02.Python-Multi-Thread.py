import threading
import time
import random

robot_telemetry = {"counter": 0}
data_lock = threading.Lock()

def read_lidar():
    for i in range(1000):
        with data_lock:
            counter = robot_telemetry["counter"]
            counter += 1
            time.sleep(0.01)
            robot_telemetry["counter"] = counter

def read_ultrasonic():
    for i in range(1000):
        with data_lock:
            counter = robot_telemetry["counter"]
            counter += 1
            time.sleep(0.01)
            robot_telemetry["counter"] = counter

if __name__ == "__main__":
    # 데몬 스레드: 메인 종료 시 함께 종료
    t1 = threading.Thread(target=read_lidar, daemon=False)
    t2 = threading.Thread(target=read_ultrasonic, daemon=False)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(robot_telemetry)

