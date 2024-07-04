import base64
import os
import queue
import sys
import threading
import time
import asyncio
import datetime
from typing import List, Tuple
import aiomysql
import cv2
import jwt
import numba
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, status, Request
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from concurrent.futures import ThreadPoolExecutor
from MysqlUtils.init import register_mysql
from RedisUtils.init import register_redis
from SmokingDecisionMaker import SmokingDecisionMaker
from importYoloPt import get_model
from utils.ConfigReader import ConfigReader
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator
from pyinstrument import Profiler
# 如果解释器在一开始就加载face_recognition可能会启动失败,这时候就需要把face_recognition放到异步方法中import,保证服务器正常启动.但这样会导致第一次检测到人脸时卡顿一下.不过放心,不会call一次导入一次的
import face_recognition
from simple_pid import PID
from line_profiler import LineProfiler

# 如果想用main.py直接启动uvicorn则需要制定一下运行时环境,要不然找到不到包和配置文件
project_root_directory = 'D:\\Smoke\\SmokeDetectionMasterPython'
os.chdir(project_root_directory)
# 小车的服务地址
base_url = "http://192.168.137.213:8000"
app = FastAPI()
jwtConfigList: List = ConfigReader().read_section("Config.ini", "Jwt")
mysqlConfigList: List = ConfigReader().read_section("Config.ini", "Mysql")
redisConfigList: List = ConfigReader().read_section("Config.ini", "Redis")
fileHomeConfigList: List = ConfigReader().read_section("Config.ini", "FileHome")
print(fileHomeConfigList)
redisPort: int = int(ConfigReader().getValueBySection(redisConfigList, "port"))
redisHost: str = ConfigReader().getValueBySection(redisConfigList, "host")
secretKey: bytes = base64.b64decode(ConfigReader().getValueBySection(jwtConfigList, "secretkey"))
algorithm: str = ConfigReader().getValueBySection(jwtConfigList, "algorithm")
DBHOST: str = ConfigReader().getValueBySection(mysqlConfigList, "host")
DBPORT: int = int(ConfigReader().getValueBySection(mysqlConfigList, "port"))
USER: str = ConfigReader().getValueBySection(mysqlConfigList, "user")
PASSWORD: str = ConfigReader().getValueBySection(mysqlConfigList, "password")
DB: str = ConfigReader().getValueBySection(mysqlConfigList, "db")
modelhome: str = ConfigReader().getValueBySection(fileHomeConfigList, "modelhome")
pichome: str = ConfigReader().getValueBySection(fileHomeConfigList, "pichome")
model, device, half, stride, names = get_model(modelhome)
register_redis(app, redisPort, redisHost)
register_mysql(app, DBHOST, DBPORT, USER, PASSWORD, DB)
IMGSZ = [512, 512]  # 图像尺寸,512这个尺寸能让cuda占用率到90+,而且准确性非常好,其实128128也可也就是检测偏移严重
IOU_THRES = 0.4  # IOU阈值
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
counts_by_territory = {}
# 异步从摄像头读取图片,避免阻塞,在单用户下无用但是多用户可用平滑帧率
executor = ThreadPoolExecutor(max_workers=4)
isAlarm = "0"


async def connect_to_pi(host, port):
    global writer
    reader, writer = await asyncio.open_connection(host, port)
    print("树莓派,启动!")
    await receive_and_parse_data(reader)


@app.on_event("startup")
async def initUserConfig():
    global userConfig
    global face_cache
    global counts_by_territory
    userConfig = {}
    async with app.state.mysql_pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("""
                SELECT t.TerritoryId, t.TerritoryName, t.ConfidenceLevel
                FROM userterritory AS ut
                LEFT JOIN territory AS t ON ut.TerritoryId = t.TerritoryId
            """)
            rows = await cur.fetchall()
            # 在初始化的时候编制索引,在运行时不需要一个个遍历寻找,查表就行.
            userConfig = {row['TerritoryId']: row for row in rows}
            face_cache = {territoryId: [] for territoryId in userConfig.keys()}
            print(userConfig)
            print(face_cache)
            # 查询当天数据条数

            today = datetime.date.today()
            for territoryId in userConfig.keys():
                # Query today's count
                await cur.execute("""
                    SELECT COUNT(*)
                    FROM smokingrecord
                    WHERE TerritoryId = %s AND DATE(SmokeStartTime) = %s
                """, (territoryId, today))
                today_count = await cur.fetchone()
                print(today_count)
                # 查询当月数据条数
                await cur.execute("""
                    SELECT COUNT(*)
                    FROM smokingrecord
                    WHERE TerritoryId = %s AND MONTH(SmokeStartTime) = MONTH(NOW())
                """, (territoryId,))
                month_count = await cur.fetchone()
                print(month_count)

                counts_by_territory[territoryId] = [today_count['COUNT(*)'], month_count['COUNT(*)']]

            print(counts_by_territory)


frame_queue_cap1 = queue.Queue(maxsize=1)
frame_queue_cap0 = queue.Queue(maxsize=1)


def camera_producer(camera, frame_queue):
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame from camera.")
            continue
        if not frame_queue.full():
            frame_queue.put(frame)


def start_camera_threads():
    cap1 = cv2.VideoCapture(1)
    cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap0.set(cv2.CAP_PROP_EXPOSURE, -6)

    threading.Thread(target=camera_producer, args=(cap1, frame_queue_cap1), daemon=True).start()
    threading.Thread(target=camera_producer, args=(cap0, frame_queue_cap0), daemon=True).start()


start_camera_threads()


# 当系统启动时候先尝试连接socket
# @app.on_event("startup")
# async def startup_event():
#     await connect_to_pi(host=Host, port=Port)
@app.on_event("startup")
async def startup_event():
    task = asyncio.create_task(connect_to_pi(host=Host, port=Port))
    print("连接任务已在后台启动")


# 异步socket写入函数
async def send_control_command(direction, angle):
    global writer
    if writer is None:
        print("写入管道不存在")
        return
    # 自定义传输规则,使用转义符分割
    command = f"{direction}/{angle}\n"
    writer.write(command.encode())
    await writer.drain()


# 自定义标记窗颜色,由于rgb->bgr的存在,所以反着写
def colors(index, bright=True):
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    return color_list[index % len(color_list)]


def do_profile(func):
    def profiled_func(*args, **kwargs):
        profiler = LineProfiler()
        try:
            profiler.add_function(func)
            profiler.enable_by_count()
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable_by_count()
            profiler.print_stats()
            sys.stdout.flush()

    return profiled_func


# 异步yolo推理主函数
async def pred_img_optimized_async(img0, model, device, imgsz, names, territoryId, iou_thres, half,
                                   line_thickness, hide_labels, hide_conf, max_det):
    loop = asyncio.get_event_loop()

    # @do_profile
    def inference():
        conf_thres = userConfig.get(territoryId).get('ConfidenceLevel')
        img = letterbox(img0, new_shape=imgsz, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
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
                # 检查类别是否为smoking,输出层"剪枝"
                if c == 0:
                    detections.append({'coords': tuple(map(int, xyxy)), 'confidence': conf.item()})
                    label = None if hide_labels else f'{names[c]} {conf:.2f}' if not hide_conf else names[c]
                    annotator.box_label(xyxy, label, color=colors(c))

        return annotator.result(), detections

    return await loop.run_in_executor(executor, inference)


# http://192.168.137.213:8080/?action=stream"

# jwt解析判断是否登陆过期/篡改
async def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, secretKey, algorithms=[algorithm])
        return payload
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token拉黑",
        )


face_data_map = {}
MAX_FACE_DATA_PER_ID = 20

EXPIRE_TIME_SECONDS = 900  # 15分钟


class FaceData:
    def __init__(self, encoding, timestamp):
        self.encoding = encoding
        self.timestamp = timestamp


async def saveImageWithPath(territoryId: int, confidence: float) -> tuple[str, str]:
    territory_dir = os.path.join(pichome, str(territoryId))
    os.makedirs(territory_dir, exist_ok=True)

    # 当前年份
    year = str(time.localtime().tm_year)
    year_dir = os.path.join(territory_dir, year)
    os.makedirs(year_dir, exist_ok=True)

    # 当前日期
    month = str(time.localtime().tm_mon)
    day = str(time.localtime().tm_mday)
    date_dir = os.path.join(year_dir, f"{month}-{day}")
    os.makedirs(date_dir, exist_ok=True)

    # 图片文件名
    file_name = f"smoke_face_{territoryId}_{confidence}_{int(time.time())}.jpg"
    return os.path.join(date_dir, file_name), file_name


async def face_distance_async(territoryId: int, frame, confidence, userid: int):
    global face_data_map

    face_encodings = face_recognition.face_encodings(frame)
    if not face_encodings:
        print("在当前帧中没有检测到人脸")
        return

    face_encoding = face_encodings[0]
    face_list = face_data_map.get(territoryId, [])

    # 逐一比对当前帧的人脸与已知的人脸
    for index, existing_face in enumerate(face_list):
        match = face_recognition.compare_faces([existing_face.encoding], face_encoding, tolerance=0.8)
        if match[0]:
            print(f"辖区 {territoryId}: 找到了相同的人脸, 索引 {index}")
            return

    # 如果没有找到相同的人脸，则添加到栈中
    current_time = time.time()
    if len(face_list) >= MAX_FACE_DATA_PER_ID:
        expired_faces = [face for face in face_list if current_time - face.timestamp > EXPIRE_TIME_SECONDS]
        for expired_face in expired_faces:
            face_list.remove(expired_face)
        if len(face_list) >= MAX_FACE_DATA_PER_ID:
            face_list.pop()  # 如果队列仍然满足最大容量，则弹出最早的数据
    face_list.append(FaceData(face_encoding, current_time))
    face_data_map[territoryId] = face_list
    print(f"辖区 {territoryId}: 检测到新的人脸, 已添加到队列")

    path, picname = await saveImageWithPath(territoryId, confidence)
    if path is not None:
        current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.imwrite(path, frame)
        async with app.state.mysql_pool.acquire() as conn:
            async with conn.cursor() as cur:
                # 执行插入操作
                await cur.execute(
                    """
                    INSERT INTO screenshotrecord (UserId, TerritoryId, ScreenshotName, ScreenshotPath, IsImportant)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (userid, territoryId, picname, path, "false"),
                )

                # 获取自增长主键的值
                auto_increment_id = cur.lastrowid

                # 插入吸烟记录
                sql = """
                INSERT INTO smokingrecord (TerritoryId,SmokeStartTime, ConfidenceLevel, ScreenshotRecordId)
                VALUES (%s, %s, %s, %s)
                """
                await cur.execute(sql, (territoryId, current_time_str, confidence, auto_increment_id))
                await conn.commit()
                counts_by_territory[territoryId][0] += 1
                counts_by_territory[territoryId][1] += 1


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


# # "http://192.168.137.213:8080/?action=stream"
# cap1 = cv2.VideoCapture("http://192.168.137.213:8080/?action=stream")
# # 如果不加入cv2.CAP_DSHOW则设置分辨率会有偏移
# cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
# cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap0.set(cv2.CAP_PROP_EXPOSURE, -6)
# fpsL = cap1.get(cv2.CAP_PROP_FPS)
# print(f"摄像头1帧率: {fpsL} FPS")
# fpsL2 = cap0.get(cv2.CAP_PROP_FPS)
# print(f"摄像头0帧率: {fpsL2} FPS")
angle_differences = []
average_window = 4


async def capture_frame(camera):
    loop = asyncio.get_event_loop()
    ret, frame = await loop.run_in_executor(executor, camera.read)
    return ret, frame


# 处理websocket的主函数
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str, territoryId: int):
    loop = asyncio.get_event_loop()

    decision_maker = SmokingDecisionMaker(confirm_threshold=20)
    # 使用 await 获取异步函数的返回值
    if await is_blacklisted(token):
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
        frame_queue = frame_queue_cap1 if territoryId == 9 else frame_queue_cap0
        # camera = cap1 if territoryId == 9 else cap0
        # PID 参数
        pid = PID(0.1, 0.002, 0.01, setpoint=0)
        while True:
            if frame_queue.empty():
                await asyncio.sleep(0.01)  # 短暂休眠以减少CPU负载
                continue

            frame = frame_queue.get()
            # ret, frame = await capture_frame(camera)
            # if not ret:
            #     print("Failed to capture frame")
            #     await websocket.close(code=1009)
            #     return

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
                angle_diff_x = dx * angle_per_pixel
                control = pid(angle_diff_x)
                if -0.8 <= control <= 0.8:
                    pass
                else:
                    direction = "left" if control < 0 else "right"
                    turn_angle = abs(control)
                    await send_control_command(direction, turn_angle)
            # if data:
            #     # 如果检测到了丁真行为则结合窗口坐标,原始图片大小,视角fov,压缩后图片大小进行计算中心点应该旋转多少度才能抵达中心
            #     x1, y1, x2, y2 = data[0]['coords']
            #     target_center_x = (x1 + x2) / 2  #
            #     img_center_x = IMGSZ[0] / 2
            #     angle_per_pixel = 50 / IMGSZ[0]
            #     dx = target_center_x - img_center_x
            #     angle_diff_x = int(dx * angle_per_pixel)
            #     # 如果接近中心则不做任何行动,避免抽搐
            #     if -2 <= angle_diff_x <= 2:
            #         pass
            #     else:
            #         # 设计一个滑动窗口,平滑每一次的移动,一秒钟在我的rtx2050上可以单用户推理15fps+,如果15+次舵机转动指令直接给树莓派他反应不过来,于是做了一下平滑处理
            #         angle_differences.append(angle_diff_x)
            #         if len(angle_differences) > average_window:
            #             angle_differences.pop(0)
            #         if len(angle_differences) == average_window:
            #             avg_angle_diff_x = sum(angle_differences) // average_window
            #             direction = "left" if avg_angle_diff_x > 0 else "right"
            #             turn_angle = abs(avg_angle_diff_x)
            #             await send_control_command(direction, turn_angle)
            #             angle_differences.clear()
            # 设计了容错,如果连续抽烟15fps以上才算你抽烟,但实际上我想到了一个更好的方法但是这里地方太小我写不下了
            decision_maker.update(data)
            if decision_maker.is_smoking_confirmed():
                print("抽烟行为发生")
                # 异步计算两帧之间的人脸相似度,如果是一个人就认为再连续抽烟,原则上只需报警一次,当然了这个协程我也不需要她的返回值,所以不用等
                asyncio.create_task(face_distance_async(territoryId, frame, data[0]['confidence'], user_id))

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
                if fps < 27:
                    print(f"当前{territoryId}辖区帧率过低为: {fps:.2f}")
                    frame_count = 0
                    start_time = time.time()
    except WebSocketDisconnect:
        pass
    finally:
        profiler.stop()
        profiler.print()


@app.websocket("/wsGetData")
async def wsGetData(websocket: WebSocket, token: str, territoryId: int):
    global isAlarm
    print(token, territoryId)
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
    try:
        while True:
            await websocket.send_json({
                "day": counts_by_territory.get(territoryId, {})[0],
                "month": counts_by_territory.get(territoryId, {})[1],
                "cache": len(face_data_map.get(territoryId, [])),
                "alarm": str(isAlarm)
            })
            if isAlarm=="1":
                isAlarm = "0"
            await asyncio.sleep(2)

    except asyncio.CancelledError:
        print("WebSocket关闭")
    except Exception as e:
        print(e)
        await websocket.close()


def is_localhost(request: Request):
    client_host = request.client.host
    return client_host in ["127.0.0.1", "::1"]


@app.get("/changeUserTerritoryConfidenceLevel")
async def changeUserTerritoryConfidenceLevel(request: Request, territoryId: int, confidenceLevel: float):
    if not is_localhost(request):
        raise HTTPException(status_code=403, detail="仅限回环地址访问")
    print("微服务请求到来")
    global userConfig
    if territoryId in userConfig:
        userConfig[territoryId]['ConfidenceLevel'] = confidenceLevel
        print(f"修改{territoryId}辖区的置信度为{confidenceLevel}")
        return {"success": "true", "message": "修改成功"}
    else:
        await initUserConfig()
        if territoryId in userConfig:
            userConfig[territoryId]['ConfidenceLevel'] = confidenceLevel
            print(f"更新{territoryId}辖区的置信度为{confidenceLevel}")
            return {"success": "true", "message": "更新成功"}
        else:
            return {"success": "false", "message": "辖区不存在"}


async def receive_and_parse_data(reader):
    global isAlarm
    try:
        while True:
            data = await reader.readuntil(separator=b'/n')  # 读取数据直到遇到分隔符
            if data:
                message = data.decode().strip('/n')  # 解码并移除分隔符
                print("接收到数据:", message)
                parts = message.split('/')
                if len(parts) > 1 and parts[1] == '1':
                    isAlarm = "1"
                    print("警报激活")
            else:
                print("连接被关闭")
                break
    except asyncio.IncompleteReadError:
        print("数据读取未完成，连接可能被关闭")
    except asyncio.CancelledError:
        print("接收数据任务被取消")
    except Exception as e:
        print(f"接收数据时发生错误: {e}")
# if __name__ == "__main__":
#     uvicorn.run("Controller.main:app", host="0.0.0.0", port=8000)
