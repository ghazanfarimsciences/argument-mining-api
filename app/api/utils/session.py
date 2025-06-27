import uuid

_active_sessions = set()

def ensure_session(session_id: str) -> str:
    """
    If session_id is blank or unknown, generate a new one.
    Otherwise return the given one.
    """
    if not session_id or session_id not in _active_sessions:
        session_id = uuid.uuid4().hex
        _active_sessions.add(session_id)
    return session_id
