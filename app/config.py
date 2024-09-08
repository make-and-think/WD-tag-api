import logging
import sys
from inspect import currentframe

from dynaconf import Dynaconf
from loguru import logger

MODEL_MAPPING = {
    "SWINV2_MODEL_DSV3_REPO": "SmilingWolf/wd-swinv2-tagger-v3",
    "CONV_MODEL_DSV3_REPO": "SmilingWolf/wd-convnext-tagger-v3",
    "VIT_MODEL_DSV3_REPO": "SmilingWolf/wd-vit-tagger-v3",
    "VIT_LARGE_MODEL_DSV3_REPO": "SmilingWolf/wd-vit-large-tagger-v3",
    "EVA02_LARGE_MODEL_DSV3_REPO": "SmilingWolf/wd-eva02-large-tagger-v3"
}

values = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)

# Remove the existing logger
logger.remove()
# Add a new logger with the given settings
logger.add(
    sys.stdout,
    colorize=values.get("logging.colorize"),
    level=values.get("logging.level"),
    format=values.get("logging.format"),
)
# Add logging to file
logger.add(
    values.get("logging.file_path"),
    level=values.get("logging.level"),
    format=values.get("logging.format"),
    rotation=values.get("logging.rotation"),
    enqueue=values.get("logging.enqueue"),
)
logger.level("INFO", color="<blue>")

logger.info("Starting...")
logger.info("Configuration Initialization...")

tokens = values.get("auth.tokens")

model_name = values.get("models.default")

if model_name not in MODEL_MAPPING:
    logger.error(f"Model {model_name} is not available")
    raise ValueError(f"Model {model_name} is not available")

model_repo = MODEL_MAPPING[model_name]
allow_all_images = values.get("models.allow_all_images")
execution_provider = values.get("models.execution_provider")
