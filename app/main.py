from fastapi import FastAPI
from .routers import wd_handler
from .internal.Interrogator import Interrogator, SWINV2_MODEL_DSV3_REPO

app = FastAPI()

app.include_router(wd_handler.router)

@app.on_event("startup")
def on_startup():
    # Предварительная загрузка модели
    interrogator = Interrogator()
    interrogator.load_model(SWINV2_MODEL_DSV3_REPO)
