from typing import Annotated

from fastapi import Header, HTTPException, Depends
from fastapi.security import APIKeyHeader

from .config import tokens, auth_tokens

header_scheme = APIKeyHeader(name="x-key")


async def get_token_header(x_token=Depends(header_scheme)):
    if x_token not in auth_tokens:
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def auth_token(user_token: Annotated[str | None, Header()]):
    """dependable for request"""
    return user_token in tokens
    # use value from settings
    # TODO later use database like row by id or move to web server condition
