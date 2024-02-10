import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from mss import mss
from fastapi.security import OAuth2PasswordBearer
import jwt

app = FastAPI()

SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
# OAuth2PasswordBearer 用于从请求中获取令牌
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# JWT 鉴权依赖
def jwt_authentication(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    screen_bounds = {'top': 0, 'left': 0, 'width': 192, 'height': 108}
    with mss() as sct:
        while True:
            screen_shot = sct.grab(screen_bounds)
            img = cv2.cvtColor(np.array(screen_shot), cv2.COLOR_BGRA2BGR)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            await websocket.send_bytes(frame)
