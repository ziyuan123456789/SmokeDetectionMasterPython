import asyncio
import base64
import time
from pyexpat import model
from typing import List, Any
import heartrate
import cv2
import jwt
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, HTTPException, status
from starlette.responses import JSONResponse

from RedisUtils.init import register_redis
from SmokingDecisionMaker import SmokingDecisionMaker
from configList import *
from importYoloPt import get_model
from utils.ConfigReader import ConfigReader
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator

model, device, half, stride, names = get_model()
imgsz = check_img_size([640, 640], s=stride)
app = FastAPI()
jwtConfigList: List = ConfigReader().read_section("Config.ini", "Jwt")
redisConfigList: List = ConfigReader().read_section("Config.ini", "Redis")
redisPort: int = int(ConfigReader().getValueBySection(redisConfigList, "port"))
redisHost: str = ConfigReader().getValueBySection(redisConfigList, "host")
secretKey: bytes = base64.b64decode(ConfigReader().getValueBySection(jwtConfigList, "secretkey"))
algorithm: str = ConfigReader().getValueBySection(jwtConfigList, "algorithm")
register_redis(app, redisPort, redisHost)
print(jwtConfigList)
print(algorithm)
WEIGHTS = 'weights/yolov5n.pt'
IMGSZ = [640, 640]  # 图像尺寸
CONF_THRES = 0.5  # 置信度阈值
IOU_THRES = 0.2  # IOU阈值
MAX_DET = 1000  # 最大检测数量
LINE_THICKNESS = 1  # 线条厚度
HIDE_CONF = False  # 是否隐藏置信度
HIDE_LABELS = None  # 是否隐藏标签
IMGHOME = 'C:/Users/31391/Desktop/vueserver/images/'  # 图像保存路径

cache=[]
def colors(index, bright=True):
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    return color_list[index % len(color_list)]


async def pred_img_optimized_async(img0, model, device, imgsz, names, conf_thres, iou_thres, half,
                                   line_thickness, hide_labels, hide_conf, max_det):
    loop = asyncio.get_event_loop()

    def inference():

        img = letterbox(img0, new_shape=imgsz, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # 半精度推理如果可用
        img /= 255.0  # 归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        im0 = img0.copy()
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        # 进行模型推理
        pred = model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)

        detections = []
        det = pred[0]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # Class index
                detections.append({'coords': tuple(map(int, xyxy)), 'confidence': conf.item()})
                label = None if hide_labels else f'{names[c]} {conf:.2f}' if not hide_conf else names[c]
                annotator.box_label(xyxy, label, color=colors(c))

        return annotator.result(), detections

    return await loop.run_in_executor(None, inference)


# http://192.168.50.1:8080/?action=stream"


async def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, secretKey, algorithms=[algorithm])
        return payload
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT token",
        )


decision_maker = SmokingDecisionMaker(confirm_threshold=45)

cap = cv2.VideoCapture(0)
fpsL = cap.get(cv2.CAP_PROP_FPS)
print(f"摄像头帧率: {fpsL} FPS")


# TEST_IMAGE_PATH = "1.jpg"
# test_image = cv2.imread(TEST_IMAGE_PATH)

async def capture_frame():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, cap.read)


# async def capture_frame():
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(None, lambda: (True, test_image))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str):
    global cache
    try:
        payload = await decode_jwt(token)
    except Exception as e:
        print("鉴权失败")
        await websocket.close(code=1008)
        return
    await websocket.accept()

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = await capture_frame()
        if not ret:
            print("Failed to capture frame")
            break

        start_time = time.time()
        processed_frame, data = await pred_img_optimized_async(frame, model, device, IMGSZ, names, CONF_THRES,
                                                               IOU_THRES, half, LINE_THICKNESS, HIDE_LABELS, HIDE_CONF,
                                                               MAX_DET)
        # end_time = time.time()
        # inference_time_ms = (end_time - start_time) * 1000
        # print(f"{inference_time_ms:.2f} 毫秒")
        decision_maker.update(data)
        if decision_maker.is_smoking_confirmed():
            print("抽烟行为发生")
            cache.append(processed_frame)
            if len(cache)>=2:
                gray1 = cv2.cvtColor(cache[0], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(cache[1], cv2.COLOR_BGR2GRAY)
                difference = cv2.absdiff(gray1, gray2)
                euclidean_distance = np.linalg.norm(difference)
                print("Euclidean Distance: ", euclidean_distance)
                cache=[]

            decision_maker.reset()
        ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            print("Failed to encode frame")
            continue
        await websocket.send_bytes(buffer.tobytes())

        frame_count += 1
        if (time.time() - start_time) >= 1:
            fps = frame_count / (time.time() - start_time)
            if fps < 30:
                print(f"Average FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()


@app.get("/store_and_fetch")
async def store_and_fetch(value: str):
    key = "test_key"
    timestamp = int(time.time())
    try:
        await app.state.redis.set(key, value)
        stored_value = await app.state.redis.get(key)
        result = f"{stored_value}-{timestamp}"
        return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
