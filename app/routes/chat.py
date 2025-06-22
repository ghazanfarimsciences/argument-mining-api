from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatError
from app.services import preprocessor, model_client
from app.utils.session import ensure_session
import json
router = APIRouter()

@router.post(
    "/send",
    response_class=StreamingResponse,
    responses={ 400: {"model": ChatError} },
)
async def send_chat(
    payload: ChatRequest,
):
    """
    1) Validate/create session
    2) Clean text
    3) Forward to external ML endpoint
    4) Stream back the PNG
    """
    # (a) ensure session is valid (or create if blank)
    session_id = ensure_session(payload.session_id)

    # (b) preprocess the text
    cleaned = preprocessor.clean_text(payload.message)
    
    return StreamingResponse(
        json.dumps({
            "message": cleaned,
            "session_id": session_id
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
