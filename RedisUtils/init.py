import aioredis
from fastapi import FastAPI
import aiomysql


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
        await app.state.redis.close()
