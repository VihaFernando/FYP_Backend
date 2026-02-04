from pydantic import BaseModel
from typing import List

class ProposalRequest(BaseModel):
    proposal_text: str

class SupervisorResult(BaseModel):
    supervisor_id: str
    matched_domains: List[str]
    score: float
    reason: str
