from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    """
    Simple liveness/readiness probe.
    """
    return {"status": "ok"}
