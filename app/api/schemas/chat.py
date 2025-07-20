from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    model: str
    message: str

class ChatError(BaseModel):  
    detail: str
