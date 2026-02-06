from fastapi import APIRouter

from ..task_manager import task_manager
from .deps import get_services

router = APIRouter()


@router.get("/status")
async def get_service_status():
    services = get_services()
    status = task_manager.get_service_status()
    status["inference_ready"] = bool(services.inference_service and services.inference_service.is_running)
    return status


@router.get("/metadata")
async def get_service_metadata():
    services = get_services()
    assert services.inference_service is not None, "Inference service is not initialized"
    return services.inference_service.server_metadata()

