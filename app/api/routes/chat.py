from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from api.schemas.chat import ChatRequest, ChatError
from api.services import preprocessor
from api.utils.session import ensure_session
import json

router = APIRouter()

ALLOWED_MODELS = {"modernbert", "openai", "tinyllama"}

@router.post(
    "/send",
    response_class=StreamingResponse,
    responses={400: {"model": ChatError}},
)
async def send_chat(
    payload: ChatRequest,
):
    # Custom validation for model
    if payload.model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not available"
        )
    # Custom validation for message
    if not payload.message or not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message is required"
        )

    # (a) ensure session is valid (or create if blank)
    session_id = ensure_session(payload.session_id)

    # (b) preprocess the text
    model = payload.model
    cleaned = preprocessor.clean_text(payload.message)
    
    return StreamingResponse(
        json.dumps({
            "message": cleaned,
            "session_id": session_id,
            "model": model
        }),
        media_type="application/json"
    )

#    # (c) call external model server
#    try:
#        ml_response = await model_client.render_diagram({
#            "session_id": session_id,
#            "text": cleaned
#        })
#    except Exception as e:
#        raise HTTPException(500, detail=f"Model service error: {e}")
#
#    # (d) stream binary image back
#    return StreamingResponse(
#        ml_response.aiter_bytes(),
#        media_type="image/png"
#    )
