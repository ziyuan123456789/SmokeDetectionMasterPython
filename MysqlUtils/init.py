from fastapi import FastAPI
import aiomysql


def register_mysql(app: FastAPI, host: str, port: int, user: str, password: str, db: str):
    @app.on_event("startup")
    async def startup_event() -> None:
        app.state.mysql_pool = await aiomysql.create_pool(
            host=host,
            port=port,
            user=user,
            password=password,
            db=db,
            charset='utf8mb4',
            autocommit=True,
            maxsize=10
        )
        print(f"MySQL数据库于端口{port}启动")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        app.state.mysql_pool.close()
        await app.state.mysql_pool.wait_closed()
        print("MySQL数据库连接池已关闭")
