import asyncio
import io

import xxhash

from app.config import process_pool


async def calculate_image_hash(image_bytes: io.BytesIO) -> int:
    image_bytes = image_bytes.read()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(process_pool, _cacl_hash,
                                      image_bytes)


def _cacl_hash(image_bytes: bytes) -> int:
    return xxhash.xxh32(image_bytes).intdigest()
