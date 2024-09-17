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
    "EVA02_LARGE_MODEL_DSV3_REPO": "SmilingWolf/wd-eva02-large-tagger-v3",
    "SWINV2_MODEL_DSV3_REPO_Q8": "Th3ro/wd-swinv2-tagger-v3-Q8",
    "SWINV2_MODEL_DSV3_REPO_Q4": "Th3ro/swinv2-tagger-v3-Q4"
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

# Model config
model_repo = MODEL_MAPPING[model_name]
allow_all_images = values.get("models.allow_all_images")
execution_provider = values.get("models.execution_provider")
process_pool_quantity = values.get("models.process_pool_quantity")
onnx_thread_quantity = values.get("models.onnx_thread_quantity")

# Auth config
auth_tokens = values.get('auth.tokens')
