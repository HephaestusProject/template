from typing import Any, Callable

from fastapi import FastAPI


def create_app(handler: Callable, request_type: Any, response_type: Any):

    app = FastAPI()

    @app.get("/hello")
    async def hello() -> str:
        return "hi"

    @app.post("/model")
    async def inference(request: request_type) -> response_type:
        response = handler(request)
        return response

    return app
