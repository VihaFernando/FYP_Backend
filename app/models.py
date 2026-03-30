from pydantic import BaseModel
from typing import List, Optional

class PreferredBonusWeights(BaseModel):
    publication: float
    scoreboard: float
    preferred: float


class FallbackWeights(BaseModel):
    publication: float
    scoreboard: float


class ScoringConfig(BaseModel):
    preferred_bonus: PreferredBonusWeights
    fallback: FallbackWeights


class ProposalRequest(BaseModel):
    proposal_text: str
    scoring_config: Optional[ScoringConfig] = None

class SupervisorResult(BaseModel):
    supervisor_id: str
    matched_domains: List[str]
    score: float
    reason: str


# ============================================================================
# PUBLICATION MODELS
# ============================================================================

class PublicationRequest(BaseModel):
    """Single publication submission"""
    title: str
    abstract: str
    domain_of_publication: str  # comma-separated
    year: int
    declared_interests: Optional[str] = ""


class BulkPublicationData(BaseModel):
    """Single publication in bulk request"""
    supervisor_id: str
    title: str
    abstract: str
    domain_of_publication: str
    year: int
    declared_interests: Optional[str] = ""


class BulkPublicationRequest(BaseModel):
    """Bulk publication submission"""
    publications: List[BulkPublicationData]


# ============================================================================
# SUPERVISOR LOGIN MODEL
# ============================================================================

class SupervisorLoginRequest(BaseModel):
    """Supervisor login request"""
    supervisor_id: str
    password: str


# ============================================================================
# PREFERRED PUBLICATION MODELS
# ============================================================================

class PreferredPublicationRequest(BaseModel):
    """Preferred publication submission"""
    title: str
    abstract: str


class SupervisorStatusRequest(BaseModel):
    """Request to update supervisor status"""
    is_full: bool
