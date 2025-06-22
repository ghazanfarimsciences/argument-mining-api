from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatError(BaseModel):  
    detail: str
