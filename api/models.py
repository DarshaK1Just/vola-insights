from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class RunRequest(BaseModel):
    user_id: str
    prompt: str

class UserInfo(BaseModel):
    user_id: str
    user_name: str

class RunResponse(BaseModel):
    user_name: Optional[str] = None
    response: str
    data_summary: Dict[str, Any] = {}
    visualizations: List[str] = []
    cache_hit: bool = False
    latency_ms: float = 0.0
    guardrail_flags: List[str] = []
    error: Optional[str] = None
