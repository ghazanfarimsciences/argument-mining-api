import httpx
from config import settings

async def render_diagram(payload: dict) -> httpx.Response:
    async with httpx.AsyncClient(timeout=settings.TIMEOUT_SECONDS) as client:
        resp = await client.post(settings.MODEL_API_URL, json=payload)
        resp.raise_for_status()
        return resp
