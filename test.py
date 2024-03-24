# import face_recognition
#
# # 加载第一张照片并识别面部特征
# image1 = face_recognition.load_image_file("C:/Users/wsh/Pictures/Camera Roll/f1.jpg")
# encoding1 = face_recognition.face_encodings(image1)[0]
#
# # 加载第二张照片并识别面部特征
# image2 = face_recognition.load_image_file("C:/Users/wsh/Pictures/Camera Roll/f2.jpg")
# encoding2 = face_recognition.face_encodings(image2)[0]
#
# # 比较两个面部特征
# results = face_recognition.compare_faces([encoding1], encoding2)
# print(results)
# if results[0]:
#     print("这两张照片是同一个人。")
# else:
#     print("这两张照片不是同一个人。")

# import face_recognition
# import cv2
# import time
#
# video_capture = cv2.VideoCapture(0)
#
# frame_count = 0
# start_time = time.time()
#
# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break
#
#     # 将图像转换为灰度
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # 缩小图像尺寸以加快处理速度
#     resized_frame = cv2.resize(gray_frame, (0, 0), fx=0.5, fy=0.5)
#
#     # 由于face_recognition需要彩色图像，我们需要在处理前将其转换回BGR
#     processed_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
#
#     # 使用CNN模型进行面部检测
#     face_location = face_recognition.face_locations(processed_frame, model="cnn")
#
#     for (top, right, bottom, left) in face_location:
#         # 缩放检测框以适应原始帧大小
#         top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#     # 计算FPS
#     frame_count += 1
#     elapsed_time = time.time() - start_time
#     if elapsed_time > 0:
#         fps = frame_count / elapsed_time
#         cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     cv2.imshow('image', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()
import time
from pyexpat import model
from typing import List, Any
import asyncio
import aiomysql
import cv2
import jwt
import numpy as np
import torch
from starlette.responses import JSONResponse

from MysqlUtils.init import register_mysql
from RedisUtils.init import register_redis
from SmokingDecisionMaker import SmokingDecisionMaker
from configList import *
from importYoloPt import get_model
from utils.ConfigReader import ConfigReader
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator

WEIGHTS = 'weights/yolov5n.pt'
IMGSZ = [640, 640]  # 图像尺寸
CONF_THRES = 0.5  # 置信度阈值
IOU_THRES = 0.2  # IOU阈值
MAX_DET = 1000  # 最大检测数量
LINE_THICKNESS = 1  # 线条厚度
HIDE_CONF = False  # 是否隐藏置信度
HIDE_LABELS = None  # 是否隐藏标签
model, device, half, stride, names = get_model()
imgsz = check_img_size([640, 640], s=stride)


def colors(index, bright=True):
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    return color_list[index % len(color_list)]


def pred_img_optimized_async(img0, model, device, imgsz, names, conf_thres, iou_thres, half,
                             line_thickness, hide_labels, hide_conf, max_det):
    img = letterbox(img0, new_shape=imgsz, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

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
            detections.append({'coords': tuple(map(int, xyxy)), 'confidence': conf.item()})

            label = None if hide_labels else f'{names[c]} {conf:.2f}' if not hide_conf else names[c]
            annotator.box_label(xyxy, label, color=colors(c))

    return im0, detections


cap = cv2.VideoCapture(0)

# 用于计算FPS的变量
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 对当前帧进行处理和预测
    img, detections = pred_img_optimized_async(frame, model, device, IMGSZ, names, CONF_THRES,
                                               IOU_THRES, half, LINE_THICKNESS, HIDE_LABELS, HIDE_CONF,
                                               MAX_DET)

    # 显示处理后的图像
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # 将FPS文本写入图像
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示处理后的图像
    cv2.imshow('Image', img)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()