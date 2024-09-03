import os
import glob

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from ..internal.Interrogator import Interrogator, SWINV2_MODEL_DSV3_REPO
from ..dependencies import auth_token
import io
import numpy as np
from wand.image import Image

router = APIRouter(prefix="/wd_tagger")


# We recive image from numpy
# this make bots not fast api backend:
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L50
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L66
# This what get this api as image. only numpy-compatible bytes
# https://github.com/Taruu/nude-check-tests/blob/286f1c7b12cecd5b26efbd59897f383d9cce0402/wdv3_jax_worker.py#L291


def is_square_webp(image_io: io.BytesIO) -> bool:
    try:
        with Image(blob=image_io.getvalue()) as img:
            return img.format == 'WEBP' and img.width == img.height
    except Exception:
        return False


def convert_to_square_webp(image_io: io.BytesIO, target_size: int = 1024, quality: int = 100) -> io.BytesIO:
    with Image(blob=image_io.getvalue()) as img:
        max_size = max(img.width, img.height)
        with Image(width=max_size, height=max_size, background='white') as new_img:
            # Paste original image to the center of the white square
            paste_x = (max_size - img.width) // 2
            paste_y = (max_size - img.height) // 2
            new_img.composite(img, left=paste_x, top=paste_y)
            
            # Resize to target size
            new_img.resize(target_size, target_size, filter='cubic')
            
            new_img.format = 'webp'
            new_img.compression_quality = quality
            
            output_io = io.BytesIO()
            new_img.save(file=output_io)
            output_io.seek(0)
            return output_io


def get_interrogator():
    interrogator = Interrogator()
    interrogator.load_model(SWINV2_MODEL_DSV3_REPO)  # TODO load from config by name
    return interrogator


async def read_image_as_bytesio(image: UploadFile) -> io.BytesIO:
    content = await image.read()
    return io.BytesIO(content)


@router.put("/rating")
async def return_rating(
        image: UploadFile = File(...),
        interrogator: Interrogator = Depends(get_interrogator)
):
    image_io = await read_image_as_bytesio(image)

    if not is_square_webp(image_io):
        raise HTTPException(status_code=400, detail="Only square WEBP images are allowed")

    ratings, _, _ = interrogator.predict(image_io, general_thresh=0.35, character_thresh=0.35)
    return {"ratings": {rating: float(score) for rating, score in ratings}}


@router.put("/tags")
async def return_tags(
        image: UploadFile = File(...),
        interrogator: Interrogator = Depends(get_interrogator)
):
    image_io = await read_image_as_bytesio(image)

    if not is_square_webp(image_io):
        raise HTTPException(status_code=400, detail="Only square WEBP images are allowed")

    _, general_tags, character_tags = interrogator.predict(image_io, general_thresh=0.35, character_thresh=0.35)
    return {
        "general_tags": {tag: float(score) for tag, score in general_tags},
    }


@router.put("/all")
async def return_all(
        image: UploadFile = File(...),
        interrogator: Interrogator = Depends(get_interrogator)
):
    image_io = await read_image_as_bytesio(image)

    if not is_square_webp(image_io):
        raise HTTPException(status_code=400, detail="Only square WEBP images are allowed")

    ratings, general_tags, character_tags = interrogator.predict(image_io, general_thresh=0.35, character_thresh=0.35)
    return {
        "ratings": {rating: float(score) for rating, score in ratings},
        "general_tags": {tag: float(score) for tag, score in general_tags},
    }


@router.put("/debug_convert")
async def debug_convert(image: UploadFile = File(...)):
    image_io = await read_image_as_bytesio(image)
    converted_io = convert_to_square_webp(image_io)
    return StreamingResponse(converted_io, media_type="image/webp")
