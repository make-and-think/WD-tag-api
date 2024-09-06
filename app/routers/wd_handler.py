import os
import glob

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


def is_square_webp_and_prepare(image_io: io.BytesIO, interrogator: Interrogator) -> Union[np.ndarray, bool]:
    try:
        with Image(blob=image_io.getvalue()) as img:
            is_webp = img.format == 'WEBP'
            
            # Check if the input image is square
            width, height = img.size
            is_square = width == height

        if is_webp and is_square:
            prepared_image = interrogator.prepare_image(image_io)
            return prepared_image
        else:
            return False
    except Exception:
        return False


def convert_to_square_webp(image_io: io.BytesIO, target_size: int = 1024, quality: int = 100) -> io.BytesIO:
    # TODO call form is_square_webp if in config enable
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
    prepared_image = is_square_webp_and_prepare(image_io, interrogator)
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
    prepared_image = is_square_webp_and_prepare(image_io, interrogator)
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
    prepared_image = is_square_webp_and_prepare(image_io, interrogator)
    if not isinstance(prepared_image, np.ndarray):
        raise HTTPException(status_code=400, detail="Image must be square and in WebP format")
    
    ratings, general_tags, character_tags = interrogator.predict(prepared_image, general_thresh=0.35, character_thresh=0.35)
    return {
        "ratings": {rating: float(score) for rating, score in ratings},
        "general_tags": {tag: float(score) for tag, score in general_tags},
    }

@router.put("/debug_convert")
async def debug_convert(image: UploadFile = File(...)):
    image_io = await read_image_as_bytesio(image)
    converted_io = convert_to_square_webp(image_io)
    return StreamingResponse(converted_io, media_type="image/webp")
