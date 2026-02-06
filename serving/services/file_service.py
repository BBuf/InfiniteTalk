import asyncio
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from loguru import logger


class FileService:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.input_image_dir = cache_dir / "inputs" / "imgs"
        self.input_audio_dir = cache_dir / "inputs" / "audios"
        self.output_video_dir = cache_dir / "outputs"

        self._http_client = None
        self._client_lock = asyncio.Lock()

        self.max_retries = 3
        self.retry_delay = 1.0
        self.max_retry_delay = 10.0

        for directory in [self.input_image_dir, self.output_video_dir, self.input_audio_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    async def _get_http_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)
                limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0)
                self._http_client = httpx.AsyncClient(verify=False, timeout=timeout, limits=limits, follow_redirects=True)
            return self._http_client

    async def _download_with_retry(self, url: str, max_retries: Optional[int] = None) -> httpx.Response:
        if max_retries is None:
            max_retries = self.max_retries

        last_exception = None
        retry_delay = self.retry_delay

        for attempt in range(max_retries):
            try:
                client = await self._get_http_client()
                response = await client.get(url)
                if response.status_code == 200:
                    return response
                if response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code} for {url}, attempt {attempt + 1}/{max_retries}")
                    last_exception = httpx.HTTPStatusError(
                        f"Server returned {response.status_code}", request=response.request, response=response
                    )
                else:
                    raise httpx.HTTPStatusError(f"Client error {response.status_code}", request=response.request, response=response)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"Connection error for {url}, attempt {attempt + 1}/{max_retries}: {str(e)}")
                last_exception = e
            except httpx.HTTPStatusError as e:
                if e.response and e.response.status_code < 500:
                    raise
                last_exception = e
            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {str(e)}")
                last_exception = e

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

        error_msg = f"All {max_retries} connection attempts failed for {url}"
        if last_exception:
            error_msg += f": {str(last_exception)}"
        raise httpx.ConnectError(error_msg)

    async def download_media(self, url: str, media_type: str = "image") -> Path:
        """
        Download input media (image/video/audio) to cache.

        Notes:
        - `media_type="image"` is used for both image and video conditioning to stay LightX2V-compatible.
        - If the URL has no filename extension, we fall back based on content-type.
        """
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL format: {url}")

        response = await self._download_with_retry(url)

        media_name = Path(parsed_url.path).name
        content_type = (response.headers.get("content-type") or "").lower()

        if not media_name:
            if media_type == "audio":
                default_ext = "mp3"
            else:
                # image OR video: infer from content-type
                if content_type.startswith("video/"):
                    default_ext = "mp4"
                else:
                    default_ext = "jpg"
            media_name = f"{uuid.uuid4()}.{default_ext}"

        target_dir = self.input_image_dir if media_type != "audio" else self.input_audio_dir
        media_path = target_dir / media_name
        media_path.parent.mkdir(parents=True, exist_ok=True)

        with open(media_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Successfully downloaded {media_type} from {url} to {media_path}")
        return media_path

    async def download_image(self, image_url: str) -> Path:
        return await self.download_media(image_url, "image")

    async def download_audio(self, audio_url: str) -> Path:
        return await self.download_media(audio_url, "audio")

    def get_output_path(self, save_result_path: str) -> Path:
        video_path = Path(save_result_path)
        if not video_path.is_absolute():
            return self.output_video_dir / save_result_path
        return video_path

    async def cleanup(self):
        async with self._client_lock:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()
                self._http_client = None

