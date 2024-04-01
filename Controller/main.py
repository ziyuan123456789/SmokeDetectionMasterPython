import asyncio
import base64
import os
import time
from pyexpat import model
from typing import List, Any, Callable
import asyncio
import aiomysql
import cv2
import jwt
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, status, Request
from line_profiler import profile
from pyinstrument.renderers import SpeedscopeRenderer, HTMLRenderer
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from concurrent.futures import ThreadPoolExecutor
from MysqlUtils.init import register_mysql
from RedisUtils.init import register_redis
from SmokingDecisionMaker import SmokingDecisionMaker
from configList import *
from importYoloPt import get_model
from utils.ConfigReader import ConfigReader
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator
from pyinstrument import Profiler

project_root_directory = 'D:\\Smoke\\SmokeDetectionMasterPython'
os.chdir(project_root_directory)
# 小车的服务地址
base_url = "http://192.168.137.213:8000"
model, device, half, stride, names = get_model()
app = FastAPI()
jwtConfigList: List = ConfigReader().read_section("Config.ini", "Jwt")
mysqlConfigList: List = ConfigReader().read_section("Config.ini", "Mysql")
redisConfigList: List = ConfigReader().read_section("Config.ini", "Redis")
redisPort: int = int(ConfigReader().getValueBySection(redisConfigList, "port"))
redisHost: str = ConfigReader().getValueBySection(redisConfigList, "host")
secretKey: bytes = base64.b64decode(ConfigReader().getValueBySection(jwtConfigList, "secretkey"))
algorithm: str = ConfigReader().getValueBySection(jwtConfigList, "algorithm")
DBHOST: str = ConfigReader().getValueBySection(mysqlConfigList, "host")
DBPORT: int = int(ConfigReader().getValueBySection(mysqlConfigList, "port"))
USER: str = ConfigReader().getValueBySection(mysqlConfigList, "user")
PASSWORD: str = ConfigReader().getValueBySection(mysqlConfigList, "password")
DB: str = ConfigReader().getValueBySection(mysqlConfigList, "db")
register_redis(app, redisPort, redisHost)
register_mysql(app, DBHOST, DBPORT, USER, PASSWORD, DB)
IMGSZ = [256, 256]  # 图像尺寸
CONF_THRES = 0.5  # 置信度阈值
IOU_THRES = 0.2  # IOU阈值
MAX_DET = 1000  # 最大检测数量
LINE_THICKNESS = 1  # 线条厚度
HIDE_CONF = False  # 是否隐藏置信度
HIDE_LABELS = None  # 是否隐藏标签
# 其实一开始用的就是socket,当时是因为不知道http与socket有什么差别,导致"粘包"问题的出现,后来切换到了fastapi做小车服务器,但是树莓派性能差,每秒钟
# 20+的http的消耗处理不了,于是又切换回了socket,并使用asyncio进一步优化,现在完全可用
Host = '192.168.137.213'
Port = 8000

writer = None

userConfig = {}
face_cache = {}


async def connect_to_pi(host, port):
    global writer
    _, writer = await asyncio.open_connection(host, port)
    print("树莓派,启动!")


@app.on_event("startup")
async def initUserConfig():
    global userConfig
    global face_cache
    userConfig = {}
    async with app.state.mysql_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("""
                SELECT t.TerritoryId, t.TerritoryName, t.ConfidenceLevel
                FROM userterritory AS ut
                LEFT JOIN territory AS t ON ut.TerritoryId = t.TerritoryId
            """)
            rows = await cur.fetchall()
            # 在初始化的时候编制索引,在运行时不需要一个个遍历寻找,查表就行.但是gpt写的这个代码可读性让人作呕
            userConfig = {row['TerritoryId']: row for row in rows}
            face_cache = {territoryId: [] for territoryId in userConfig.keys()}
            print(userConfig)
            print(face_cache)


# 当系统启动时候先尝试连接socket
# @app.on_event("startup")
# async def startup_event():
#     await connect_to_pi(host=Host, port=Port)


# 异步socket写入函数
async def send_control_command(direction, angle):
    global writer
    if writer is None:
        print("写入管道不存在")
        return

    command = f"{direction}/{angle}\n"
    writer.write(command.encode())
    await writer.drain()


# 自定义标记窗颜色,由于rgb->bgr的存在,所以反着写
def colors(index, bright=True):
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    return color_list[index % len(color_list)]


# 异步yolo推理主函数
async def pred_img_optimized_async(img0, model, device, imgsz, names, territoryId, iou_thres, half,
                                   line_thickness, hide_labels, hide_conf, max_det):
    loop = asyncio.get_event_loop()

    def inference():
        conf_thres = userConfig.get(territoryId).get('ConfidenceLevel')
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
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0], agnostic=False, max_det=max_det)

        detections = []
        det = pred[0]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                if c == 0:  # 检查类别是否为smoking,输出层"剪枝"
                    detections.append({'coords': tuple(map(int, xyxy)), 'confidence': conf.item()})
                    label = None if hide_labels else f'{names[c]} {conf:.2f}' if not hide_conf else names[c]
                    annotator.box_label(xyxy, label, color=colors(c))

        return annotator.result(), detections

    return await loop.run_in_executor(None, inference)


# http://192.168.137.213:8080/?action=stream"

# jwt解析判断是否登陆过期,后续加入鉴权
async def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, secretKey, algorithms=[algorithm])
        return payload
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token拉黑",
        )


# 异步从摄像头读取图片,避免阻塞,在单用户下无用但是多用户可用平滑帧率
executor = ThreadPoolExecutor(max_workers=6)


async def capture_frame0():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, cap0.read)


async def capture_frame1():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, cap1.read)


face_data_map = {}
MAX_FACE_DATA_PER_ID = 20


async def face_distance_async(territoryId: int, frame):
    import face_recognition
    rgb_frame = frame[:, :, ::-1]
    face_encodings = face_recognition.face_encodings(frame)
    if not face_encodings:
        file_name = f"no_face_{territoryId}_{int(time.time())}.jpg"
        file_path = os.path.join("D:/Smoke/PictureHome/9/2023/1-25", file_name)

        # 保存图像
        cv2.imwrite(file_path, frame)
        print("在当前帧中没有检测到人脸")

        return

    face_encoding = face_encodings[0]
    face_list = face_data_map.get(territoryId, [])

    # 逐一比对当前帧的人脸与已知的人脸
    for index, existing_face in enumerate(face_list):
        match = face_recognition.compare_faces([existing_face], face_encoding, tolerance=0.6)
        if match[0]:
            print(f"辖区 {territoryId}: 找到了相同的人脸, 索引 {index}")
            return

    # 如果没有找到相同的人脸，则添加到队列中
    if len(face_list) >= MAX_FACE_DATA_PER_ID:
        face_list.pop(0)  # 如果队列已满，则删除最早的数据
    face_list.append(face_encoding)
    face_data_map[territoryId] = face_list
    print(f"辖区 {territoryId}: 检测到新的人脸, 已添加到队列")


async def is_blacklisted(jwt_id: str) -> bool:
    """
    检查JWT ID是否被拉黑（即存在于Redis中）。
    """
    exists = await app.state.redis.exists(jwt_id)
    return exists == 1  # 如果存在，返回True


async def is_user_authorized(user_id: int, territoryId: int) -> bool:
    async with app.state.mysql_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("SELECT TerritoryId FROM userterritory WHERE UserId = %s and TerritoryId=%s",
                              (user_id, territoryId))
            result = await cur.fetchone()
            print(result)
            return result is None


# "http://192.168.137.213:8080/?action=stream"
cap1 = cv2.VideoCapture(1)
cap1.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap0 = cv2.VideoCapture(0)
cap0.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
fpsL = cap1.get(cv2.CAP_PROP_FPS)
print(f"摄像头1帧率: {fpsL} FPS")
fpsL2 = cap0.get(cv2.CAP_PROP_FPS)
print(f"摄像头0帧率: {fpsL2} FPS")
angle_differences = []
average_window = 7


# 处理websocket的主函数
@profile
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str, territoryId: int):
    decision_maker = SmokingDecisionMaker(confirm_threshold=20)
    if await is_blacklisted(token):  # 使用 await 获取异步函数的返回值
        print("jwt被拉黑")
        await websocket.close(code=1008)
        return
    try:
        payload = await decode_jwt(token)
        user_id = payload.get('id')
        role = payload.get('role')
        if role == '0':
            if await is_user_authorized(user_id, territoryId):
                await websocket.close(code=1008)
                print("用户越权访问其他辖区")
                return
    except Exception as e:
        print("鉴权失败")
        await websocket.close(code=1008)
        return

    await websocket.accept()

    frame_count = 0
    start_time = time.time()
    profiler = Profiler()
    try:
        profiler.start()
        while True:
            # 其实应该在连接时确认而不是检测时确认,但是在这个方法里还不能先定义在后面await,我也不知道为什么.不过比较一个变量的消耗可忽略不计
            if territoryId == 9:
                ret, frame = await capture_frame1()
            else:
                ret, frame = await capture_frame0()
            if not ret:
                print("Failed to capture frame")
                await websocket.close(code=1009)
                return

            processed_frame, data = await pred_img_optimized_async(frame, model, device, IMGSZ, names, territoryId,
                                                                   IOU_THRES, half, LINE_THICKNESS, HIDE_LABELS,
                                                                   HIDE_CONF,
                                                                   MAX_DET)
            if data:
                # 如果检测到了丁真行为则结合窗口坐标,原始图片大小,视角fov,压缩后图片大小进行计算中心点应该旋转多少度才能抵达中心
                x1, y1, x2, y2 = data[0]['coords']
                target_center_x = (x1 + x2) / 2  #
                img_center_x = IMGSZ[0] / 2
                angle_per_pixel = 50 / IMGSZ[0]
                dx = target_center_x - img_center_x
                angle_diff_x = int(dx * angle_per_pixel)
                # 如果接近中心则不做任何行动,避免抽搐
                if -2 <= angle_diff_x <= 2:
                    pass
                else:
                    # 设计一个滑动窗口,平滑每一次的移动,一秒钟在我的rtx2050上可以单用户推理15fps+,如果15+次舵机转动指令直接给树莓派他反应不过来,于是做了一下平滑处理
                    angle_differences.append(angle_diff_x)
                    if len(angle_differences) > average_window:
                        angle_differences.pop(0)
                    if len(angle_differences) == average_window:
                        avg_angle_diff_x = sum(angle_differences) // average_window
                        direction = "left" if avg_angle_diff_x > 0 else "right"
                        turn_angle = abs(avg_angle_diff_x)
                        # await send_control_command(direction, turn_angle)
                        angle_differences.clear()
            # 设计了容错,如果连续抽烟15fps以上才算你抽烟,但实际上我想到了一个更好的方法但是这里地方太小我写不下了
            decision_maker.update(data)
            if decision_maker.is_smoking_confirmed():
                print("抽烟行为发生")
                # 异步计算两帧之间的人脸相似度,如果是一个人就认为再连续抽烟,原则上只需报警一次,当然了这个协程我也不需要她的返回值,所以不用等
                asyncio.create_task(face_distance_async(territoryId, frame))

                decision_maker.reset()
            # 我尝试使用ffmpeg+nginx做推流,但是性能更差,最后还是选择了websocket做法,避免每次的http请求的线程创建/请求头消耗
            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                print("Failed to encode frame")
                continue

            await websocket.send_bytes(buffer.tobytes())

            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                if fps < 7:
                    print(f"当前{territoryId}辖区帧率过低为: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
    except WebSocketDisconnect:
        pass
    finally:
        profiler.stop()
        profiler.print()


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


@app.get("/items")
async def read_items():
    async with app.state.mysql_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("SELECT * FROM user")
            result = await cur.fetchall()
            return result


def is_localhost(request: Request):
    client_host = request.client.host
    return client_host in ["127.0.0.1", "::1"]


@app.get("/changeUserTerritoryConfidenceLevel")
async def changeUserTerritoryConfidenceLevel(request: Request, territoryId: int, confidenceLevel: float):
    if not is_localhost(request):
        raise HTTPException(status_code=403, detail="仅回环地址访问")

    global userConfig
    if territoryId in userConfig:
        userConfig[territoryId]['ConfidenceLevel'] = confidenceLevel
        return {"message": "Confidence level updated successfully"}
    else:
        await initUserConfig()
        if territoryId in userConfig:
            userConfig[territoryId]['ConfidenceLevel'] = confidenceLevel
            return {"message": "Confidence level updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Territory ID not found")


if __name__ == "__main__":
    uvicorn.run("Controller.main:app", host="0.0.0.0", port=8000)
