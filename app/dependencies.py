from typing import Annotated

from fastapi import Header
from .config import tokens


async def auth_token(user_token: Annotated[str | None, Header()]):
    """dependable for request"""
    return user_token in tokens
    # use value from settings
    # TODO later use database like row by id or move to web server condition
