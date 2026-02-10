import random
from typing import Optional

from pydantic import BaseModel, Field

from .utils.generate_task_id import generate_task_id


def generate_random_seed() -> int:
    return random.randint(0, 2**32 - 1)


class TalkObject(BaseModel):
    audio: str = Field(..., description="Audio path")
    mask: str = Field(..., description="Mask path")


class BaseTaskRequest(BaseModel):
    # Keep field names identical to LightX2V for compatibility.
    task_id: str = Field(default_factory=generate_task_id, description="Task ID (auto-generated)")
    prompt: str = Field("", description="Generation prompt")
    use_prompt_enhancer: bool = Field(False, description="Whether to use prompt enhancer")
    negative_prompt: str = Field("", description="Negative prompt")

    # NOTE: In InfiniteTalk, this path can be either an image OR a video.
    image_path: str = Field("", description="Base64 encoded image/video, URL, or local path")

    save_result_path: str = Field("", description="Save result path (optional, defaults to task_id, suffix auto-detected)")
    infer_steps: int = Field(40, description="Inference steps (maps to InfiniteTalk sampling_steps)")
    seed: int = Field(default_factory=generate_random_seed, description="Random seed (auto-generated if not set)")

    target_shape: list[int] = Field([], description="Return video or image shape")
    lora_name: Optional[str] = Field(None, description="(compat) LoRA filename, ignored by InfiniteTalk serving")
    lora_strength: float = Field(1.0, description="(compat) LoRA strength, ignored by InfiniteTalk serving")

    # InfiniteTalk-specific optional knobs (added without breaking compatibility).
    size: Optional[str] = Field(None, description="infinitetalk-480 or infinitetalk-720 (optional override)")
    motion_frame: Optional[int] = Field(None, description="Motion frame length for streaming continuation")
    text_guide_scale: Optional[float] = Field(None, description="Text CFG scale override")
    audio_guide_scale: Optional[float] = Field(None, description="Audio CFG scale override")
    sample_shift: Optional[float] = Field(None, description="Noise schedule shift override")
    max_frame_num: Optional[int] = Field(None, description="Max frames for streaming generation override")
    use_teacache: Optional[bool] = Field(None, description="Enable TeaCache (optional override)")
    teacache_thresh: Optional[float] = Field(None, description="TeaCache threshold (optional override)")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.save_result_path:
            self.save_result_path = f"{self.task_id}"

    def get(self, key, default=None):
        return getattr(self, key, default)


class VideoTaskRequest(BaseTaskRequest):
    # Keep fields identical to LightX2V.
    num_fragments: int = Field(1, description="Number of fragments")
    target_video_length: int = Field(81, description="Target video length (frames)")
    audio_path: str = Field("", description="Input audio path")
    video_duration: int = Field(5, description="(compat) Video duration")
    talk_objects: Optional[list[TalkObject]] = Field(None, description="(compat) Talk objects")
    target_fps: Optional[int] = Field(25, description="Target FPS (InfiniteTalk default is 25)")
    resize_mode: Optional[str] = Field("adaptive", description="(compat) Resize mode")


class ImageTaskRequest(BaseTaskRequest):
    aspect_ratio: str = Field("16:9", description="(compat) Output aspect ratio")


class TaskStatusMessage(BaseModel):
    task_id: str = Field(..., description="Task ID")


class TaskResponse(BaseModel):
    task_id: str
    task_status: str
    save_result_path: str


class StopTaskResponse(BaseModel):
    stop_status: str
    reason: str

