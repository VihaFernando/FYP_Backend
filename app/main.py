from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from app.data_loader import load_scoreboard
from app.models import (
    ProposalRequest, 
    PublicationRequest, 
    BulkPublicationRequest,
    SupervisorLoginRequest,
    PreferredPublicationRequest,
    SupervisorStatusRequest
)
from app.recommender import recommend_supervisors
from app.mongodb_handler import (
    connect_mongodb,
    insert_publication,
    bulk_insert_publications,
    get_supervisor_scoreboard,
    reload_scoreboard_from_mongodb,
    auth_supervisor,
    get_supervisor_publications,
    get_supervisor_profile,
    get_db,
    update_publication,
    add_preferred_publication,
    update_preferred_publication,
    delete_preferred_publication,
    delete_publication as delete_pub_handler,
    get_supervisor_full_status,
    toggle_supervisor_full_status,
    get_full_supervisors,
    PUBLICATIONS_COLLECTION,
    SCOREBOARD_COLLECTION,
    _serialize_docs
)
from app.publication_pipeline import process_new_publications
import pandas as pd

app = FastAPI(title="Supervisor Research Recommender")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB on startup
@app.on_event("startup")
async def startup_event():
    result = connect_mongodb()
    if result is not None:
        print("✅ Application started with MongoDB connection")
    else:
        print("✅ Application started in offline mode (CSV only, MongoDB unavailable)")

# Load initial scoreboard
scoreboard = load_scoreboard()

# ============================================================================
# RECOMMENDATION ENDPOINT (Existing)
# ============================================================================

@app.post("/recommend")
def recommend(req: ProposalRequest):
    # Try to load from MongoDB first (latest data)
    database = get_db()
    if database is not None:
        try:
            scoreboard_docs = list(database[SCOREBOARD_COLLECTION].find({}))
            if scoreboard_docs:
                # Serialize MongoDB docs
                scoreboard_docs = _serialize_docs(scoreboard_docs)
                scoreboard = pd.DataFrame(scoreboard_docs)
                print(f"Using MongoDB scoreboard ({len(scoreboard)} entries) for recommendations")
            else:
                # Fallback to CSV if MongoDB is empty
                scoreboard = load_scoreboard()
                print("MongoDB scoreboard empty, using CSV fallback")
        except Exception as e:
            print(f"Error loading from MongoDB: {e}, using CSV fallback")
            scoreboard = load_scoreboard()
    else:
        # Offline mode - use CSV
        scoreboard = load_scoreboard()
        print("Using CSV scoreboard for recommendations (MongoDB offline)")
    
    try:
        results = recommend_supervisors(
            req.proposal_text,
            scoreboard,
            database=database,
            scoring_config=req.scoring_config
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "proposal": req.proposal_text,
        "recommendations": results[:5]
    }


# ============================================================================
# PUBLICATION MANAGEMENT ENDPOINTS (NEW)
# ============================================================================

@app.post("/publications/add")
async def add_publication(req: PublicationRequest):
    """
    Add a single publication and automatically update the scoreboard.
    
    Request body:
    {
        "supervisor_id": "SUP-001",
        "title": "Publication Title",
        "abstract": "Publication abstract...",
        "domain_of_publication": "AI, NLP",
        "year": 2024,
        "declared_interests": "AI, NLP, LLM"  # (optional)
    }
    """
    result = insert_publication(
        supervisor_id=req.supervisor_id,
        title=req.title,
        abstract=req.abstract,
        domain_of_publication=req.domain_of_publication,
        year=req.year,
        declared_interests=req.declared_interests or ""
    )
    
    # Reload scoreboard into memory
    global scoreboard
    try:
        scoreboard = load_scoreboard()
    except:
        pass
    
    return result


@app.post("/publications/bulk")
async def add_bulk_publications(req: BulkPublicationRequest):
    """
    Add multiple publications in one operation and update the scoreboard.
    
    Request body:
    {
        "publications": [
            {
                "supervisor_id": "SUP-001",
                "title": "Title 1",
                "abstract": "Abstract 1",
                "domain_of_publication": "AI",
                "year": 2024,
                "declared_interests": "AI"
            },
            ...
        ]
    }
    """
    result = bulk_insert_publications(req.publications)
    
    # Reload scoreboard into memory
    global scoreboard
    try:
        scoreboard = load_scoreboard()
    except:
        pass
    
    return result


@app.get("/scoreboard/supervisor/{supervisor_id}")
async def get_supervisor_scores(supervisor_id: str):
    """Get all domain scores for a specific supervisor"""
    scores = get_supervisor_scoreboard(supervisor_id)
    return {
        "supervisor_id": supervisor_id,
        "domains": scores,
        "total_domains": len(scores)
    }


@app.post("/scoreboard/rebuild")
async def rebuild_scoreboard():
    """
    Rebuild the entire scoreboard from MongoDB publications.
    WARNING: This recalculates everything from scratch.
    """
    result = reload_scoreboard_from_mongodb()
    
    # Reload scoreboard into memory
    global scoreboard
    try:
        scoreboard = load_scoreboard()
    except:
        pass
    
    return result


# ============================================================================
# SUPERVISOR ENDPOINTS
# ============================================================================

@app.post("/supervisor/auth")
async def login_supervisor(req: SupervisorLoginRequest):
    """
    Supervisor login with shared password and supervisor ID.
    
    Request body:
    {
        "supervisor_id": "SUP-001",
        "password": "<shared_password>"
    }
    """
    result = auth_supervisor(req.supervisor_id, req.password)
    return result


@app.get("/supervisor/{supervisor_id}/profile")
async def get_profile(supervisor_id: str):
    """Get supervisor's complete profile (publications + scoreboard)"""
    return get_supervisor_profile(supervisor_id)


@app.get("/supervisor/{supervisor_id}/full-status")
async def get_full_status(supervisor_id: str):
    """Get the full/available status of a supervisor"""
    return get_supervisor_full_status(supervisor_id)


@app.post("/supervisor/{supervisor_id}/full-status")
async def update_full_status(supervisor_id: str, req: SupervisorStatusRequest):
    """Toggle supervisor's full/available status"""
    return toggle_supervisor_full_status(supervisor_id, req.is_full)


@app.post("/supervisor/{supervisor_id}/publication/add")
async def add_supervisor_publication(supervisor_id: str, req: PublicationRequest):
    """
    Add a new publication for supervisor.
    Automatically updates scoreboard.
    """
    result = insert_publication(
        supervisor_id=supervisor_id,
        title=req.title,
        abstract=req.abstract,
        domain_of_publication=req.domain_of_publication,
        year=req.year,
        declared_interests=req.declared_interests or ""
    )
    
    # Reload scoreboard into memory
    global scoreboard
    try:
        scoreboard = load_scoreboard()
    except:
        pass
    
    return result


# ============================================================================
# PUBLICATION DELETION
# ============================================================================

@app.delete("/supervisor/{supervisor_id}/publication/{publication_id}")
async def delete_publication_endpoint(supervisor_id: str, publication_id: str):
    """Delete a publication and update scoreboard"""
    result = delete_pub_handler(supervisor_id, publication_id)
    return result


# ============================================================================
# PUBLICATION UPDATE
# ============================================================================

@app.put("/supervisor/{supervisor_id}/publication/{publication_id}")
async def update_publication_endpoint(supervisor_id: str, publication_id: str, req: PublicationRequest):
    """Update an existing publication and recalculate scoreboard"""
    result = update_publication(
        supervisor_id=supervisor_id,
        publication_id=publication_id,
        title=req.title,
        abstract=req.abstract,
        domain_of_publication=req.domain_of_publication,
        year=req.year,
        declared_interests=req.declared_interests or ""
    )
    return result


# ============================================================================
# PREFERRED PUBLICATION CRUD
# ============================================================================

@app.post("/supervisor/{supervisor_id}/preferred-publication/add")
async def add_preferred_publication_endpoint(supervisor_id: str, req: PreferredPublicationRequest):
    """Add preferred publication with embedding generation."""
    return add_preferred_publication(
        supervisor_id=supervisor_id,
        title=req.title,
        abstract=req.abstract
    )


@app.put("/supervisor/{supervisor_id}/preferred-publication/{publication_id}")
async def update_preferred_publication_endpoint(
    supervisor_id: str,
    publication_id: str,
    req: PreferredPublicationRequest
):
    """Update preferred publication and regenerate embedding from latest text."""
    return update_preferred_publication(
        supervisor_id=supervisor_id,
        publication_id=publication_id,
        title=req.title,
        abstract=req.abstract
    )


@app.delete("/supervisor/{supervisor_id}/preferred-publication/{publication_id}")
async def delete_preferred_publication_endpoint(supervisor_id: str, publication_id: str):
    """Delete preferred publication by publication_id."""
    return delete_preferred_publication(
        supervisor_id=supervisor_id,
        publication_id=publication_id
    )


# ============================================================================
# INTERESTS MANAGEMENT
# ============================================================================

@app.put("/supervisor/{supervisor_id}/interests")
async def update_interests(supervisor_id: str, req: dict):
    """Update supervisor's declared interests"""
    try:
        database = get_db()
        if not database:
            return {"status": "error", "message": "Database not connected"}
        
        interests_str = req.get("declared_interests", "")
        interests_list = [i.strip() for i in interests_str.split(",") if i.strip()]
        
        # Update all publications for this supervisor with new interests
        database[PUBLICATIONS_COLLECTION].update_many(
            {"supervisor_id": supervisor_id},
            {"$set": {"declared_interests": interests_list}}
        )
        
        return {
            "status": "success",
            "message": "Interests updated",
            "interests": interests_list
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to update interests: {str(e)}"
        }


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "scoreboard_loaded": scoreboard is not None,
        "scoreboard_size": len(scoreboard) if scoreboard is not None else 0
    }
