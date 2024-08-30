import os
import glob

from fastapi import APIRouter

router = APIRouter(prefix="/wd_tagger")


# We recive image from numpy
# this make bots not fast api backend:
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L50
# https://github.com/Taruu/nude-check-tests/blob/main/wdv3_jax_worker.py#L66
# This what get this api as image. only numpy-compatible bytes
# https://github.com/Taruu/nude-check-tests/blob/286f1c7b12cecd5b26efbd59897f383d9cce0402/wdv3_jax_worker.py#L291
@router.put("/rating")
async def return_rating(image: bytes):
    """Return only rating of image"""
    pass


@router.put("/tags")
async def return_rating(image: bytes):
    """Return only tags of image"""
    pass


@router.put("/all")
async def return_rating(image: bytes):
    """Return all info of image"""
    pass
