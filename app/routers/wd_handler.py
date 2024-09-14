import asyncio
import os
import glob
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor

import magic
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from ..internal.interrogator import Interrogator
from ..dependencies import auth_token
import io
import numpy as np
from wand.image import Image
from typing import Union, Tuple, Any
from ..config import model_repo, allow_all_images, process_pool_quantity, logger

def create_interrogator():
    interrogator = Interrogator()
    interrogator.load_model(model_repo)
    return interrogator


@asynccontextmanager
async def lifespan(app: APIRouter):
    # Load the ML model
    logger.info(f"Start load the ML model {model_repo}")
    yield
    # Clean up the ML models and release the resources
    logger.info("Unload model")



router = APIRouter(prefix="/wd_tagger", lifespan=lifespan)
process_pool = ProcessPoolExecutor(max_workers=process_pool_quantity)


# We recive image from numpy
# this make bots not fast api backend:
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L50
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L66
# This what get this api as image. only numpy-compatible bytes
# https://github.com/Taruu/nude-check-tests/blob/286f1c7b12cecd5b26efbd59897f383d9cce0402/wdv3_jax_worker.py#L291


def image_prepare(image_io: io.BytesIO, target_size: int) -> Union[np.ndarray, bool]:
    if not allow_all_images and magic.from_buffer(image_io.read(1024), mime=True) != "image/webp":
        raise HTTPException(status_code=400, detail="Image must be in WebP format")

    image_obj = Image(blob=image_io.getvalue())
    print(type(image_obj.size))
    width, height = image_obj.size
    print(width, type(width))
    if not allow_all_images and (width != height):
        raise HTTPException(status_code=400, detail="Image must be square")

    if allow_all_images or (image_obj.size != (target_size, target_size)):
        image_obj.resize(target_size, target_size, filter='cubic')

    if image_obj.alpha_channel:
        image_obj.alpha_channel = 'remove'

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
    interrogator = create_interrogator()
    prepared_image = image_prepare(image_file, interrogator.model_target_size)
    ratings, general_tags, character_tags = interrogator.predict(prepared_image, general_thresh=0.35,
                                                                 character_thresh=0.35)
    return ratings, general_tags, character_tags


async def image_predict(image_file: io.BytesIO) -> tuple[Any, Any, Any]:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(process_pool, _image_predict, image_file)
    except Exception as e:
        logger.error(f"Error during image prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during image processing")


@router.put("/rating")
async def return_rating(
        image: UploadFile = File(...)
):
    image_bytes = io.BytesIO(await image.read())
    ratings, _, _ = await image_predict(image_bytes)
    return {"ratings": {rating: float(score) for rating, score in ratings}}


@router.put("/tags")
async def return_tags(
        image: UploadFile = File(...)
):
    image_bytes = io.BytesIO(await image.read())
    _, general_tags, character_tags = await image_predict(image_bytes)

    return {
        "general_tags": {tag: float(score) for tag, score in general_tags},
    }


@router.put("/all")
async def return_all(
        image: UploadFile = File(...)
):
    image_bytes = io.BytesIO(await image.read())
    ratings, general_tags, character_tags = await image_predict(image_bytes)
    return {
        "ratings": {rating: float(score) for rating, score in ratings},
        "general_tags": {tag: float(score) for tag, score in general_tags},
    }
