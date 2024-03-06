import base64
import time
from pyexpat import model
from typing import List, Any

import cv2
import jwt
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, HTTPException, status
from starlette.responses import JSONResponse

from RedisUtils.init import register_redis
from configList import *
from importYoloPt import get_model
from utils.ConfigReader import ConfigReader
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors

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
CONF_THRES = 0.1  # 置信度阈值
IOU_THRES = 0.1  # IOU阈值
MAX_DET = 1000  # 最大检测数量
LINE_THICKNESS = 2  # 线条厚度
HIDE_CONF = True  # 是否隐藏置信度
HIDE_LABELS = None  # 是否隐藏标签
IMGHOME = 'C:/Users/31391/Desktop/vueserver/images/'  # 图像保存路径

def colors(index, bright=True):
    # A simple function to cycle colors based on index
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    return color_list[index % len(color_list)]


def pred_img_optimized(img0, model, device, imgsz, stride, names, conf_thres, iou_thres, half=False,
                       line_thickness=2, hide_labels=False, hide_conf=True, max_det=1000):
    # Image preprocessing
    img = letterbox(img0, new_shape=imgsz, auto=True)[0]  # Resize and pad
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # Half precision
    img = img / 255.0  # Normalize to [0, 1]
    if len(img.shape) == 3:  # Add batch dimension
        img = img[None]

    im0 = img0.copy()
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

    # Inference
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)
    det = pred[0]
    if len(det):
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # Class ID
            label = None if hide_labels else (f'{names[c]} {conf:.2f}' if not hide_conf else names[c])
            annotator.box_label(xyxy, label, color=colors(c))

    return annotator.result()

# --host="192.168.5.229"
cap = cv2.VideoCapture("http://192.168.50.1:8080/?action=stream")
async def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, secretKey, algorithms=[algorithm])
        return payload
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT token",
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str) -> Any:
    try:
        payload = await decode_jwt(token)  # 这里假设你有一个有效的JWT解码函数
    except HTTPException as e:
        print("鉴权失败")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    await websocket.accept()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break  # 如果无法获取帧，则跳出循环

        # 调用pred_img_optimized()处理帧
        processed_frame = pred_img_optimized(
            frame, model, device, IMGSZ, stride, names, CONF_THRES, IOU_THRES, half,
            LINE_THICKNESS, HIDE_LABELS, HIDE_CONF, MAX_DET
        )

        # 编码处理后的帧以便传输
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            print("Failed to encode frame")
            continue  # 如果编码失败，继续下一次循环

        frame_bytes = buffer.tobytes()
        await websocket.send_bytes(frame_bytes)


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


