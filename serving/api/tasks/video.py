import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from ...schema import TaskResponse, VideoTaskRequest
from ...task_manager import task_manager
from ..deps import get_services, validate_url_async

router = APIRouter()


def _write_file_sync(file_path: Path, content: bytes) -> None:
    with open(file_path, "wb") as buffer:
        buffer.write(content)


@router.post("/", response_model=TaskResponse)
async def create_video_task(message: VideoTaskRequest):
    try:
        # NOTE: message.image_path can be image OR video; validate URL if needed.
        if hasattr(message, "image_path") and message.image_path and message.image_path.startswith("http"):
            if not await validate_url_async(message.image_path):
                raise HTTPException(status_code=400, detail=f"Media URL is not accessible: {message.image_path}")

        # audio is required for InfiniteTalk; keep compatibility but fail early.
        if not getattr(message, "audio_path", ""):
            raise HTTPException(status_code=400, detail="audio_path is required for InfiniteTalk video generation")

        task_id = task_manager.create_task(message)
        message.task_id = task_id

        return TaskResponse(task_id=task_id, task_status="pending", save_result_path=message.save_result_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create video task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/form", response_model=TaskResponse)
async def create_video_task_form(
    # Keep param name `image_file` for LightX2V compatibility.
    # In InfiniteTalk serving, this file can be either an image or a video.
    image_file: UploadFile = File(...),
    prompt: str = Form(default=""),
    save_result_path: str = Form(default=""),
    use_prompt_enhancer: bool = Form(default=False),
    negative_prompt: str = Form(default=""),
    num_fragments: int = Form(default=1),
    infer_steps: int = Form(default=40),
    target_video_length: int = Form(default=81),
    seed: int = Form(default=42),
    audio_file: UploadFile = File(None),
    video_duration: int = Form(default=5),
    target_fps: int = Form(default=25),
):
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    async def save_file_async(file: UploadFile, target_dir: Path) -> str:
        if not file or not file.filename:
            return ""

        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = target_dir / unique_filename

        content = await file.read()
        await asyncio.to_thread(_write_file_sync, file_path, content)
        return str(file_path)

    media_path = ""
    if image_file and image_file.filename:
        media_path = await save_file_async(image_file, services.file_service.input_image_dir)

    audio_path = ""
    if audio_file and audio_file.filename:
        audio_path = await save_file_async(audio_file, services.file_service.input_audio_dir)

    if not media_path:
        raise HTTPException(status_code=400, detail="image_file is required (can be image or video)")
    if not audio_path:
        raise HTTPException(status_code=400, detail="audio_file is required for InfiniteTalk")

    message = VideoTaskRequest(
        prompt=prompt,
        use_prompt_enhancer=use_prompt_enhancer,
        negative_prompt=negative_prompt,
        image_path=media_path,
        num_fragments=num_fragments,
        save_result_path=save_result_path,
        infer_steps=infer_steps,
        target_video_length=target_video_length,
        seed=seed,
        audio_path=audio_path,
        video_duration=video_duration,
        target_fps=target_fps,
    )

    try:
        task_id = task_manager.create_task(message)
        message.task_id = task_id
        return TaskResponse(task_id=task_id, task_status="pending", save_result_path=message.save_result_path)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create video form task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

