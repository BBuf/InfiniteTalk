from typing import Any, Optional

from loguru import logger

from ...schema import TaskResponse
from ..file_service import FileService
from ..inference import DistributedInferenceService
from .base import BaseGenerationService


class ImageGenerationService(BaseGenerationService):
    """
    Kept for LightX2V API compatibility.

    InfiniteTalk does not implement image generation. This service will fail fast.
    """

    def __init__(self, file_service: FileService, inference_service: DistributedInferenceService):
        super().__init__(file_service, inference_service)

    def get_output_extension(self) -> str:
        return ".png"

    def get_task_type(self) -> str:
        return ""

    async def generate_with_stop_event(self, message: Any, stop_event) -> Optional[Any]:
        logger.error("Image tasks are not supported by InfiniteTalk serving")
        raise RuntimeError("Image tasks are not supported. Use /v1/tasks/video.")

    async def generate_image_with_stop_event(self, message: Any, stop_event) -> Optional[Any]:
        return await self.generate_with_stop_event(message, stop_event)

