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
from utils.plots import Annotator
import torchvision
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


class SmokingDecisionMaker:
    def __init__(self, confirm_threshold=5):
        self.confirm_threshold = confirm_threshold
        self.smoking_streak = 0
        self.smoking_confirmed = False

    def update(self, detections):
        if detections:
            self.smoking_streak += 1

        else:
            self.smoking_streak = 0
        if self.smoking_streak >= self.confirm_threshold:
            self.smoking_confirmed = True
        else:
            self.smoking_confirmed = False

    def reset(self):
        self.smoking_streak = 0
        self.smoking_confirmed = False

    def is_smoking_confirmed(self):
        return self.smoking_confirmed



def colors(index, bright=True):
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    return color_list[index % len(color_list)]


def pred_img_optimized(img0, model, device, imgsz, names, conf_thres, iou_thres, half,
                       line_thickness, hide_labels, hide_conf, max_det):
    img = letterbox(img0, new_shape=imgsz, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img = img / 255.0
    if len(img.shape) == 3:
        img = img[None]
    im0 = img0.copy()
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)
    detections = []
    det = pred[0]
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            detections.append({'coords': tuple(map(int, xyxy)), 'confidence': conf})
            if hide_labels:
                label = None
            else:
                if not hide_conf:
                    label = f'{names[c]} {conf:.2f}'
                else:
                    label = names[c]
            annotator.box_label(xyxy, label, color=colors(c))


    else:
        return img0, []

    return annotator.result(), detections


# http://192.168.50.1:8080/?action=stream"
cap = cv2.VideoCapture(0)


async def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, secretKey, algorithms=[algorithm])
        return payload
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT token",
        )

decision_maker = SmokingDecisionMaker(confirm_threshold=15)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str) -> Any:
    try:
        payload = await decode_jwt(token)
    except HTTPException as e:
        print("鉴权失败")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    await websocket.accept()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("图片处理异常")
            break

        processed_frame, data = pred_img_optimized(
            frame, model, device, IMGSZ, names, CONF_THRES, IOU_THRES, half,
            LINE_THICKNESS, HIDE_LABELS, HIDE_CONF, MAX_DET
        )
        decision_maker.update(data)
        if decision_maker.is_smoking_confirmed():
            print("吸烟行为已确认")
            decision_maker.reset()
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            print("Failed to encode frame")
            continue
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
