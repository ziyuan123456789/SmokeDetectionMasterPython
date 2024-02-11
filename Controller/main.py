import base64
import time
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from mss import mss
from fastapi.security import OAuth2PasswordBearer
import jwt
from starlette.responses import JSONResponse

from RedisUtils.init import register_redis
from Utils.ConfigReader import ConfigReader

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


async def decode_jwt(token: str):

        print(secretKey)
        payload = jwt.decode(token, secretKey, algorithms=[algorithm])
        return payload



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str):
    print(token)
    try:
        payload = await decode_jwt(token)
    except HTTPException as e:
        print("鉴权失败")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    await websocket.accept()

    screen_bounds = {'top': 0, 'left': 0, 'width': 192, 'height': 108}
    with mss() as sct:
        while True:
            screen_shot = sct.grab(screen_bounds)
            img = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_BGRA2BGR)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            await websocket.send_bytes(frame)
