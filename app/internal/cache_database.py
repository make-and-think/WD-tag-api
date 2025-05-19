import io
import json

import redis.asyncio as redis
from sympy.codegen.cnodes import union

from app.utils.images import calculate_image_hash


class ImageCacheInterface:
    def __init__(self, host="localhost", port=6379, password=None, expired_time=1800):
        self.db_obj = redis.Redis(host=host, port=port, password=password, decode_responses=True)
        self._folder_name = "wd-tag-cache"
        self._expired_time = expired_time

    async def put_image(self, image_hash_or_bytes: int | io.BytesIO, image_data: dict):
        if type(image_hash_or_bytes) is io.BytesIO:
            image_hash_or_bytes = await calculate_image_hash(image_hash_or_bytes)

        await self.db_obj.set(f"{self._folder_name}:{image_hash_or_bytes}", json.dumps(image_data),
                              ex=self._expired_time)
        return True

    async def get_image(self, image_hash: int) -> dict:
        image_json_data = await self.db_obj.get(f"{self._folder_name}:{image_hash}")
        if not image_json_data:
            return None
        return json.loads(image_json_data)
