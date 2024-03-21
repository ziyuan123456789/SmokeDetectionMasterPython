import aioredis
from fastapi import FastAPI


def register_redis(app: FastAPI, redisPort: int, redisHost: str):
    @app.on_event("startup")
    async def startup_event() -> None:
        app.state.redis = await aioredis.Redis(host=redisHost, port=redisPort, db=3, encoding="utf-8")
        print(f"redis成功--->>{app.state.redis}")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        print("redis关闭")
        await app.state.redis.close()
import aioredis
from fastapi import FastAPI


def register_redis(app: FastAPI, port: int, host: str):
    async def redis_pool():
        redis = await aioredis.from_url(
            url="redis://127.0.0.1", port=port, db=2, encoding="utf-8", decode_responses=True
        )
        print("Reids于端口" + str(port) + "启动")
        return redis

    @app.on_event("startup")
    async def srartup_event():
        app.state.redis = await redis_pool()

    @app.on_event("shutdown")
    async def shutdown_event():
        app.state.redis.close()
        await app.state.redis.wait_closed()
