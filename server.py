from pathlib import Path

from pydantic import BaseModel

from predictor import Predictor
from serving.app_factory import create_app


predictor = Predictor()


class Request(BaseModel):
    inputs: str


class Response(BaseModel):
    outputs: str


def handler(request: Request) -> Response:
    prediction = predictor.predict(request.inputs)
    return Response(outputs=prediction)


app = create_app(handler, Request, Response)
