import uuid

def ensure_session(session_id: str | None) -> str:
    """
    If session_id is empty, generate a new one.
    Otherwise, return it as-is (assume client manages session).
    """
    if not session_id or not session_id.strip():
        return uuid.uuid4().hex
    return session_id.strip()

