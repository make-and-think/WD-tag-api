import os
import glob

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from ..internal.Interrogator import Interrogator, SWINV2_MODEL_DSV3_REPO
from ..dependencies import auth_token
import io
import numpy as np

router = APIRouter(prefix="/wd_tagger")


# We recive image from numpy
# this make bots not fast api backend:
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L50
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L66
# This what get this api as image. only numpy-compatible bytes
# https://github.com/Taruu/nude-check-tests/blob/286f1c7b12cecd5b26efbd59897f383d9cce0402/wdv3_jax_worker.py#L291


# TODO allow only square webp images.
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
    ratings, _, _ = interrogator.predict(image_io, general_thresh=0.35, character_thresh=0.35)
    return {"ratings": {rating: float(score) for rating, score in ratings}}


@router.put("/tags")
async def return_tags(
        image: UploadFile = File(...),
        interrogator: Interrogator = Depends(get_interrogator)
):
    image_io = await read_image_as_bytesio(image)
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
    ratings, general_tags, character_tags = interrogator.predict(image_io, general_thresh=0.35, character_thresh=0.35)
    return {
        "ratings": {rating: float(score) for rating, score in ratings},
        "general_tags": {tag: float(score) for tag, score in general_tags},
    }
