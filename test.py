import multiprocessing as mp
import time
import cv2

def queue_img_put(q, camera_index):
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if ret:
            q.put(frame)
        else:
            break

def queue_img_get(q, camera_index):
    start_time = time.time()
    frame_count = 0

    while True:
        frame = q.get()
        frame_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:  # 每秒计算一次帧率
            fps = frame_count / elapsed_time
            print(f"Camera {camera_index}: {fps:.2f} FPS")
            frame_count = 0
            start_time = time.time()

def run_multi_camera():
    camera_indices = [0, 1]  # 假设本地摄像头索引为0和1
    mp.set_start_method('spawn')
    queues = [mp.Queue(maxsize=2) for _ in camera_indices]

    processes = []
    for camera_index, queue in zip(camera_indices, queues):
        processes.append(mp.Process(target=queue_img_put, args=(queue, camera_index)))
        processes.append(mp.Process(target=queue_img_get, args=(queue, camera_index)))

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    run_multi_camera()
