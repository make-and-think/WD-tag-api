import asyncio
from contextlib import asynccontextmanager

import magic
from PIL.Image import Resampling
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from ..internal.cache_database import ImageCacheInterface
from ..internal.onnx_interrogator import Interrogator
from ..dependencies import get_token_header
import io
import numpy as np
from PIL import Image as pilImage
from typing import Union, Any
from ..config import model_repo, allow_all_images, logger, auth_tokens, process_pool, database_host, database_port, \
    database_password
from ..utils.images import calculate_image_hash

wd_interrogator = Interrogator()


@asynccontextmanager
async def lifespan(app: APIRouter):
    # Load the ML model
    logger.info(f"Start load the ML model {model_repo}")
    app.state.interrogator = wd_interrogator
    app.state.interrogator.load_model(model_repo)
    yield
    # Clean up the ML models and release the resources
    logger.info("Unload model")
    del app.state.interrogator


dependencies_list = []
if auth_tokens:
    dependencies_list.append(Depends(get_token_header))

router = APIRouter(prefix="/wd_tagger", lifespan=lifespan, dependencies=dependencies_list)
print(database_host, database_port, database_password)
if database_host:
    database_worker = ImageCacheInterface(host=database_host, port=database_port, password=database_password)
else:
    database_worker = None


# We recive image from numpy
# this make bots not fast api backend:
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L50
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L66
# This what get this api as image. only numpy-compatible bytes
# https://github.com/Taruu/nude-check-tests/blob/286f1c7b12cecd5b26efbd59897f383d9cce0402/wdv3_jax_worker.py#L291


def image_prepare(image_io: io.BytesIO, target_size: int) -> Union[np.ndarray, bool]:
    # TODO check if another types use wandImage
    # image_obj = wandImage(blob=image_io.getvalue())
    image_obj = pilImage.open(image_io)
    width, height = image_obj.size
    if not allow_all_images and (width != height):
        raise HTTPException(status_code=400, detail="Image must be square")

    if allow_all_images or (image_obj.size != (target_size, target_size)):
        image_obj = image_obj.resize((target_size, target_size), resample=Resampling.BICUBIC)
        # image_obj.resize(target_size, target_size, filter='cubic')

    # if image_obj.alpha_channel:
    #     image_obj.alpha_channel = 'remove'

    if image_obj.mode in ('RGBA', 'LA') or (image_obj.mode == 'P' and 'transparency' in image_obj.info):
        image_obj = image_obj.convert("RGB")

    image_array = np.array(image_obj)

    # Ensure the image is in RGB format
    if image_array.shape[2] > 3:
        image_array = image_array[:, :, :3]

    # Convert RGB to BGR
    image_array = image_array[:, :, ::-1]
    array = np.expand_dims(image_array, axis=0)
    return array.astype(np.float32)


async def read_image_as_bytesio(image: UploadFile) -> io.BytesIO:
    content = await image.read()
    return io.BytesIO(content)


def _image_predict(image_file: io.BytesIO) -> tuple[Any, Any, Any]:
    """CPU bound image predict"""
    logger.debug(f"Start predict image: {hash(image_file)}")
    print("target size", wd_interrogator.model_target_size)
    prepared_image = image_prepare(image_file, wd_interrogator.model_target_size)
    ratings, general_tags, character_tags = wd_interrogator.predict(prepared_image, general_thresh=0.35,
                                                                    character_thresh=0.35)
    logger.debug(f"End predict image: {hash(image_file)}")
    return ratings, general_tags, character_tags


async def image_predict(image_file: io.BytesIO) -> dict:
    image_data = None
    image_hash = None

    if database_worker:  # TODO REPLACE WITH FUNC TO VARRRABLE
        image_hash = await calculate_image_hash(image_file)
        image_data = await database_worker.get_image(image_hash)

    if image_data:
        return image_data

    loop = asyncio.get_event_loop()

    current_mimetype = magic.from_buffer(image_file.read(1024), mime=True)
    image_file.seek(0)

    if (not allow_all_images) and (current_mimetype != "image/webp"):
        raise HTTPException(status_code=400, detail="Image must be in WebP format")

    print(current_mimetype, allow_all_images, not "image" in current_mimetype)

    if allow_all_images and (not "image" in current_mimetype):
        raise HTTPException(status_code=400, detail=f"Not are image, current: {current_mimetype}")

    prepared_image = await loop.run_in_executor(process_pool, image_prepare, image_file,
                                                wd_interrogator.model_target_size)

    image_predicted_data = await wd_interrogator.async_predict(prepared_image, general_thresh=0.35,
                                                               character_thresh=0.35)

    print(image_hash, image_data)
    image_data = {
        "ratings": {rating: float(score) for rating, score in image_predicted_data[0]},
        "general_tags": {tag: float(score) for tag, score in image_predicted_data[1]},
        "characters": {character: float(score) for character, score in image_predicted_data[2]}
    }
    if not image_hash and database_worker:
        image_hash = await calculate_image_hash(image_file)
    if database_worker:
        await database_worker.put_image(image_hash, dict(image_data))

    return image_data


@router.get("/model_image_size")
async def model_image_size():
    return wd_interrogator.model_target_info


@router.put("/rating")
async def return_rating(
        image: UploadFile = File(...)
):
    image_bytes = io.BytesIO(await image.read())
    image_prediction = await image_predict(image_bytes)

    image_prediction.pop("characters")
    image_prediction.pop("general_tags")

    return image_prediction


@router.put("/tags")
async def return_tags(
        image: UploadFile = File(...)
):
    image_bytes = io.BytesIO(await image.read())
    image_prediction = await image_predict(image_bytes)

    image_prediction.pop("characters")
    image_prediction.pop("ratings")

    return image_prediction


@router.put("/all")
async def return_all(
        image: UploadFile = File(...)
):
    image_bytes = io.BytesIO(await image.read())
    image_prediction = await image_predict(image_bytes)

    image_prediction.pop("characters")

    return image_prediction
