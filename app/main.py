from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.data_loader import load_scoreboard
from app.models import ProposalRequest
from app.recommender import recommend_supervisors

app = FastAPI(title="Supervisor Research Recommender")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scoreboard = load_scoreboard()

@app.post("/recommend")
def recommend(req: ProposalRequest):
    results = recommend_supervisors(req.proposal_text, scoreboard)
    return {
        "proposal": req.proposal_text,
        "recommendations": results[:5]
    }
