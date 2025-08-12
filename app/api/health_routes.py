from fastapi import APIRouter

from app.models.schemas import HealthCheckResponse

router = APIRouter(tags=["Health Check"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(
        status="ok"
    )
