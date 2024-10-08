from fastapi import FastAPI
from .routers import image_handler
from .internal.onnx_interrogator import Interrogator
from .config import model_repo, logger

app = FastAPI()

app.include_router(image_handler.router)


# TODO https://github.com/fastapi/fastapi/issues/608
@app.on_event("startup")
def on_startup():
    pass
