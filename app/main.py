from fastapi import FastAPI
from .routers import wd_handler
from .internal.interrogator import Interrogator
from .config import get_model_settings, logger

app = FastAPI()

app.include_router(wd_handler.router)

@app.on_event("startup")
def on_startup():
    model_settings = get_model_settings()
    model_name = model_settings["default_model"]
    
    if model_name not in model_settings["available_models"]:
        logger.error(f"Model {model_name} is not available")
        raise ValueError(f"Model {model_name} is not available")
    
    interrogator = Interrogator()
    interrogator.load_model(model_name)
    
    logger.info(f"Loaded model: {model_name}")
    
    # Сохраняем interrogator в состоянии приложения
    app.state.interrogator = interrogator