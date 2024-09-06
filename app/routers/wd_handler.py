import os
import glob

import magic
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from ..internal.interrogator import Interrogator, SWINV2_MODEL_DSV3_REPO
from ..dependencies import auth_token
import io
import numpy as np
from wand.image import Image
from typing import Union

router = APIRouter(prefix="/wd_tagger")


# We recive image from numpy
# this make bots not fast api backend:
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L50
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L66
# This what get this api as image. only numpy-compatible bytes
# https://github.com/Taruu/nude-check-tests/blob/286f1c7b12cecd5b26efbd59897f383d9cce0402/wdv3_jax_worker.py#L291


def image_prepare(image_io: io.BytesIO, target_size: int) -> Union[np.ndarray, bool]:
    if magic.from_buffer(image_io.read(1024)) != "image/webp":
        return False
    image_obj = Image(blob=image_io.getvalue())
    width, height = image_obj.size
    if width != height:
        return False
    if image_obj.size != (target_size, target_size):
        image_obj.resize(target_size, target_size, filter='cubic')

    if image_obj.alpha_channel:
        image_obj.alpha_channel = 'remove'

    image_array = np.array(image_obj)

    # Ensure the image is in RGB format
    if image_array.shape[2] > 3:
        image_array = image_array[:, :, :3]

    # Convert RGB to BGR
    image_array = image_array[:, :, ::-1]

    return np.expand_dims(image_array, axis=0).astype(np.float32)

    return prepared_image


def get_interrogator(request: Request):
    return request.app.state.interrogator


async def read_image_as_bytesio(image: UploadFile) -> io.BytesIO:
    content = await image.read()
    return io.BytesIO(content)


@router.put("/rating")
async def return_rating(
        image: UploadFile = File(...),
        interrogator: Interrogator = Depends(get_interrogator)
):
    image_io = await read_image_as_bytesio(image)
    prepared_image = image_prepare(image_io, interrogator)
    if not isinstance(prepared_image, np.ndarray):
        raise HTTPException(status_code=400, detail="Image must be square and in WebP format")

    ratings, _, _ = interrogator.predict(prepared_image, general_thresh=0.35, character_thresh=0.35)
    return {"ratings": {rating: float(score) for rating, score in ratings}}


@router.put("/tags")
async def return_tags(
        image: UploadFile = File(...),
        interrogator: Interrogator = Depends(get_interrogator)
):
    image_io = await read_image_as_bytesio(image)
    prepared_image = image_prepare(image_io, interrogator)
    if not isinstance(prepared_image, np.ndarray):
        raise HTTPException(status_code=400, detail="Image must be square and in WebP format")

    _, general_tags, character_tags = interrogator.predict(prepared_image, general_thresh=0.35, character_thresh=0.35)
    return {
        "general_tags": {tag: float(score) for tag, score in general_tags},
    }


@router.put("/all")
async def return_all(
        image: UploadFile = File(...),
        interrogator: Interrogator = Depends(get_interrogator)
):
    image_io = await read_image_as_bytesio(image)
    prepared_image = image_prepare(image_io, interrogator)
    if not isinstance(prepared_image, np.ndarray):
        raise HTTPException(status_code=400, detail="Image must be square and in WebP format")

    ratings, general_tags, character_tags = interrogator.predict(prepared_image, general_thresh=0.35,
                                                                 character_thresh=0.35)
    return {
        "ratings": {rating: float(score) for rating, score in ratings},
        "general_tags": {tag: float(score) for tag, score in general_tags},
    }


@router.put("/debug_convert")
async def debug_convert(image: UploadFile = File(...)):
    image_io = await read_image_as_bytesio(image)
    converted_io = convert_to_square_webp(image_io)
    return StreamingResponse(converted_io, media_type="image/webp")
