"""
MongoDB Handler - Manages publication insertion and scoreboard updates
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import ObjectId
from bson.decimal128 import Decimal128
import os
import math
import hmac
from datetime import datetime
from dotenv import load_dotenv
from app.publication_pipeline import process_new_publications, update_scoreboard_file
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load environment variables
load_dotenv()

# MongoDB connection - require environment variables for security
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

if not MONGODB_URI:
    raise ValueError("❌ MONGODB_URI environment variable is required and not set. Please add it to your .env file.")
if not DB_NAME:
    raise ValueError("❌ DB_NAME environment variable is required and not set. Please add it to your .env file.")
SUPERVISOR_SHARED_PASSWORD = os.getenv("SUPERVISOR_SHARED_PASSWORD")
if not SUPERVISOR_SHARED_PASSWORD:
    raise ValueError("password is not set")
PUBLICATIONS_COLLECTION = "publications"
SCOREBOARD_COLLECTION = "supervisor_scoreboard"
PREFERRED_PROJECTS_COLLECTION = "preferred_projects"
SUPERVISORS_COLLECTION = "supervisors"

client = None
db = None


def connect_mongodb():
    """Initialize MongoDB connection"""
    global client, db
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[DB_NAME]
        print(f"✅ Connected to MongoDB: {DB_NAME}")
        return db
    except Exception as e:
        print(f"⚠️  MongoDB connection failed ({type(e).__name__}). Running in offline mode (CSV only).")
        db = None
        client = None
        return None


def get_db():
    """Get MongoDB database instance"""
    global db
    if db is None:
        connect_mongodb()
    return db


def _serialize_value(value):
    # Handle NaN and Inf
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0
    # Handle Decimal128
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    # Handle ObjectId
    if isinstance(value, ObjectId):
        return str(value)
    # Handle datetime
    if isinstance(value, datetime):
        return value.isoformat()
    # Handle lists
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    # Handle dicts
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _serialize_docs(docs):
    return [_serialize_value(doc) for doc in docs]


# Embedding model cache
_embedding_model = None

def get_embedding_model():
    """Load embedding model (cached)"""
    global _embedding_model
    if _embedding_model is None:
        print("Loading sentence-transformers model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def generate_embedding(text: str):
    """Generate embedding for text"""
    if not text or not isinstance(text, str):
        return None
    try:
        model = get_embedding_model()
        return model.encode(text).tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def recalculate_scoreboard(supervisor_id: str):
    """Recalculate and update scoreboard for a supervisor"""
    try:
        database = get_db()
        if database is None:
            print(f"⚠️ Database not connected, skipping scoreboard recalculation")
            return
        
        # Always delete old scoreboard entries first
        delete_result = database[SCOREBOARD_COLLECTION].delete_many(
            {"supervisor_id": supervisor_id}
        )
        print(f"🗑️ Deleted {delete_result.deleted_count} old scoreboard entries for {supervisor_id}")
        
        # Fetch remaining publications
        publications = list(database[PUBLICATIONS_COLLECTION].find(
            {"supervisor_id": supervisor_id}
        ))
        
        # If no publications left, return (scoreboard is cleared)
        if not publications:
            print(f"ℹ️ No publications for {supervisor_id}, scoreboard cleared")
            return
        
        # Convert to plain dicts with proper field structure
        pub_data = [{
            "supervisor_id": pub.get("supervisor_id"),
            "title": pub.get("title", ""),
            "abstract": pub.get("abstract", ""),
            "domain of the publication": pub.get("domain of the publication", ""),
            "year": pub.get("year", 2026),
            "declared_interests": pub.get("declared_interests", "")
        } for pub in publications]
        
        print(f"📊 Processing {len(pub_data)} publications for {supervisor_id}")
        updated_scores = process_new_publications(pub_data)
        
        # Insert new scoreboard entries
        if not updated_scores.empty:
            updated_scores["supervisor_id"] = supervisor_id
            insert_result = database[SCOREBOARD_COLLECTION].insert_many(
                updated_scores.to_dict('records')
            )
            print(f"✅ Inserted {len(insert_result.inserted_ids)} new scoreboard entries for {supervisor_id}")
        else:
            print(f"⚠️ No scores generated for {supervisor_id}")
            
    except Exception as e:
        print(f"❌ Error recalculating scoreboard: {e}")
        import traceback
        traceback.print_exc()


def insert_publication(
    supervisor_id: str,
    title: str,
    abstract: str,
    domain_of_publication: str,
    year: int,
    declared_interests: str = ""
) -> dict:
    """
    Insert a new publication and automatically update the scoreboard.
    
    Args:
        supervisor_id: Supervisor ID (e.g., "SUP-001")
        title: Publication title
        abstract: Publication abstract
        domain_of_publication: Comma-separated domains (e.g., "AI, NLP")
        year: Publication year
        declared_interests: Supervisor's declared interests
    
    Returns:
        Result dict with status and updated scoreboard
    """
    
    database = get_db()
    
    # Create publication document
    publication = {
        "supervisor_id": supervisor_id,
        "title": title,
        "abstract": abstract,
        "domain of the publication": domain_of_publication,
        "year": year,
        "declared_interests": declared_interests,
        "created_at": datetime.utcnow()
    }
    
    # Generate embedding for the abstract
    embedding = generate_embedding(abstract)
    if embedding:
        publication["embedding"] = embedding
    
    # Insert into MongoDB (if connected)
    if database is not None:
        try:
            result = database[PUBLICATIONS_COLLECTION].insert_one(publication)
            publication["_id"] = str(result.inserted_id)
            publication["publication_id"] = str(result.inserted_id)  # Create publication_id field
            
            # Update document with publication_id
            database[PUBLICATIONS_COLLECTION].update_one(
                {"_id": result.inserted_id},
                {"$set": {"publication_id": str(result.inserted_id)}}
            )
            print(f"✅ Publication inserted: {publication['publication_id']}")
        except Exception as e:
            print(f"⚠️  Failed to insert into MongoDB: {e}")
    
    # =============================
    # TRIGGER PIPELINE UPDATE
    # =============================
    
    try:
        # Fetch all publications for this supervisor (or all if new supervisor)
        if database is not None:
            all_publications = list(database[PUBLICATIONS_COLLECTION].find({
                "supervisor_id": supervisor_id
            }))
        else:
            # Offline mode: use the single publication
            all_publications = [publication]
        
        # Convert to plain dicts
        pub_data = [{
            "supervisor_id": pub.get("supervisor_id"),
            "title": pub.get("title", ""),
            "abstract": pub.get("abstract", ""),
            "domain of the publication": pub.get("domain of the publication", ""),
            "year": pub.get("year", 2026),
            "declared_interests": pub.get("declared_interests", "")
        } for pub in all_publications]
        
        # Run pipeline
        updated_scoreboard = process_new_publications(
            pub_data,
            existing_scoreboard_path="data/final_scoreboard.csv"
        )
        
        # Save to MongoDB only (no CSV update)
        if database is not None:
            try:
                # Only delete THIS SUPERVISOR's scoreboard
                database[SCOREBOARD_COLLECTION].delete_many({"supervisor_id": supervisor_id})
                # Filter scoreboard to only include this supervisor's data
                supervisor_scores = updated_scoreboard[updated_scoreboard["supervisor_id"] == supervisor_id]
                if not supervisor_scores.empty:
                    scoreboard_records = supervisor_scores.to_dict('records')
                    database[SCOREBOARD_COLLECTION].insert_many(scoreboard_records)
                    print(f"✅ Scoreboard updated in MongoDB ({len(scoreboard_records)} records)")
                else:
                    print(f"⚠️  No scores generated for {supervisor_id}")
            except Exception as e:
                print(f"⚠️  Failed to update scoreboard in MongoDB: {e}")
        
        return {
            "status": "success",
            "message": f"Publication added and scoreboard updated for {supervisor_id}",
            "publication_id": str(publication.get("_id", "unknown"))
        }
    
    except Exception as e:
        print(f"❌ Error updating scoreboard: {e}")
        return {
            "status": "error",
            "message": f"Publication added, but scoreboard update failed: {str(e)}",
            "publication_id": str(publication.get("_id", "unknown"))
        }


def bulk_insert_publications(publications_list: list) -> dict:
    """
    Insert multiple publications and update scoreboard once.
    
    Args:
        publications_list: List of publication dicts
    
    Returns:
        Result dict
    """
    
    database = get_db()
    inserted_count = 0
    
    # Insert into MongoDB
    if database is not None:
        try:
            result = database[PUBLICATIONS_COLLECTION].insert_many(publications_list)
            inserted_count = len(result.inserted_ids)
        except Exception as e:
            print(f"⚠️  Failed to bulk insert: {e}")
    
    # =============================
    # TRIGGER PIPELINE UPDATE
    # =============================
    
    try:
        # Run pipeline on ALL publications for affected supervisors
        referenced_sups = set(pub.get("supervisor_id") for pub in publications_list)
        
        if database is not None:
            fetch_filter = {"supervisor_id": {"$in": list(referenced_sups)}}
            all_publications = list(database[PUBLICATIONS_COLLECTION].find(fetch_filter))
        else:
            all_publications = publications_list
        
        # Convert to plain dicts
        pub_data = [{
            "supervisor_id": pub.get("supervisor_id"),
            "title": pub.get("title", ""),
            "abstract": pub.get("abstract", ""),
            "domain of the publication": pub.get("domain of the publication", ""),
            "year": pub.get("year", 2026),
            "declared_interests": pub.get("declared_interests", "")
        } for pub in all_publications]
        
        # Run pipeline
        updated_scoreboard = process_new_publications(
            pub_data,
            existing_scoreboard_path="data/final_scoreboard.csv"
        )
        
        # Update in MongoDB only (no CSV update)
        if database is not None:
            try:
                # Delete scoreboard only for affected supervisors, then reinsert
                for sup_id in referenced_sups:
                    database[SCOREBOARD_COLLECTION].delete_many({"supervisor_id": sup_id})
                
                # Filter scoreboard to only include affected supervisors' data
                sup_scores = updated_scoreboard[updated_scoreboard["supervisor_id"].isin(list(referenced_sups))]
                if not sup_scores.empty:
                    scoreboard_records = sup_scores.to_dict('records')
                    database[SCOREBOARD_COLLECTION].insert_many(scoreboard_records)
                    print(f"✅ Scoreboard updated for {len(referenced_sups)} supervisors")
                else:
                    print(f"⚠️  No scores generated")
            except Exception as e:
                print(f"⚠️  Failed to update scoreboard in MongoDB: {e}")
        
        return {
            "status": "success",
            "message": f"Bulk inserted {inserted_count} publications, scoreboard updated",
            "inserted_count": inserted_count,
            "scoreboard_rows_updated": len(updated_scoreboard),
            "affected_supervisors": list(referenced_sups)
        }
    
    except Exception as e:
        print(f"❌ Error updating scoreboard during bulk insert: {e}")
        return {
            "status": "error",
            "message": str(e),
            "inserted_count": inserted_count
        }


def get_supervisor_scoreboard(supervisor_id: str) -> list:
    """Fetch scoreboard entries for a specific supervisor"""
    database = get_db()
    if database is not None:
        try:
            return list(database[SCOREBOARD_COLLECTION].find(
                {"supervisor_id": supervisor_id},
                {"_id": 0}
            ))
        except Exception as e:
            print(f"⚠️  Failed to fetch scoreboard: {e}")
    
    # Fallback to CSV
    try:
        df = pd.read_csv("data/final_scoreboard.csv")
        return df[df["supervisor_id"] == supervisor_id].to_dict('records')
    except:
        return []


def reload_scoreboard_from_mongodb() -> dict:
    """Rebuild scoreboard from all publications in MongoDB"""
    database = get_db()
    
    if database is None:
        return {"status": "error", "message": "MongoDB not connected"}
    
    try:
        all_publications = list(database[PUBLICATIONS_COLLECTION].find())
        
        if not all_publications:
            return {"status": "warning", "message": "No publications found in MongoDB"}
        
        # Convert to plain dicts
        pub_data = [{
            "supervisor_id": pub.get("supervisor_id"),
            "title": pub.get("title", ""),
            "abstract": pub.get("abstract", ""),
            "domain of the publication": pub.get("domain of the publication", ""),
            "year": pub.get("year", 2026),
            "declared_interests": pub.get("declared_interests", "")
        } for pub in all_publications]
        
        # Run pipeline
        updated_scoreboard = process_new_publications(pub_data)
        
        # Update MongoDB only (no CSV update)
        database[SCOREBOARD_COLLECTION].delete_many({})
        scoreboard_records = updated_scoreboard.to_dict('records')
        database[SCOREBOARD_COLLECTION].insert_many(scoreboard_records)
        
        return {
            "status": "success",
            "message": "Scoreboard rebuilt from MongoDB",
            "publications_processed": len(all_publications),
            "scoreboard_entries": len(updated_scoreboard)
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# SUPERVISOR-SPECIFIC FUNCTIONS
# ============================================================================

def auth_supervisor(supervisor_id: str, password: str) -> dict:
    """
    Authenticate supervisor with shared password and supervisor ID existence.
    
    Args:
        supervisor_id: Supervisor ID (e.g., "SUP-001")
    
    Returns:
        Dict with auth status
    """
    try:
        database = get_db()

        if not password or not hmac.compare_digest(password, SUPERVISOR_SHARED_PASSWORD):
            return {
                "status": "error",
                "authenticated": False,
                "message": "Invalid credentials"
            }
        
        if database is None:
            return {
                "status": "error",
                "authenticated": False,
                "message": "MongoDB connection not available"
            }
        
        # Check publications collection with flexible field names
        publications = database[PUBLICATIONS_COLLECTION]
        pub_doc = publications.find_one(
            {"$or": [
                {"supervisor_id": supervisor_id},
                {"id": supervisor_id}
            ]}
        )
        
        # Check scoreboard collection
        scoreboard = database[SCOREBOARD_COLLECTION]
        score_doc = scoreboard.find_one({"supervisor_id": supervisor_id})
        
        if pub_doc or score_doc:
            return {
                "status": "success",
                "authenticated": True,
                "supervisor_id": supervisor_id,
                "message": f"Welcome back!"
            }
        else:
            return {
                "status": "error",
                "authenticated": False,
                "message": f"Supervisor {supervisor_id} not found in system"
            }
    
    except Exception as e:
        print(f"Auth error: {type(e).__name__}: {e}")
        return {
            "status": "error",
            "authenticated": False,
            "message": f"Authentication error: {str(e)}"
        }


def update_publication(
    supervisor_id: str,
    publication_id: str,
    title: str,
    abstract: str,
    domain_of_publication: str,
    year: int,
    declared_interests: str = ""
) -> dict:
    """Update an existing publication and recalculate scoreboard"""
    try:
        database = get_db()
        if database is None:
            return {"status": "error", "message": "Database not connected"}
        
        # Generate new embedding
        embedding = generate_embedding(abstract)
        
        # Update publication by publication_id field
        update_data = {
            "title": title,
            "abstract": abstract,
            "domain of the publication": domain_of_publication,
            "year": year,
            "declared_interests": declared_interests,
            "updated_at": datetime.utcnow()
        }
        if embedding:
            update_data["embedding"] = embedding
        
        result = database[PUBLICATIONS_COLLECTION].update_one(
            {"supervisor_id": supervisor_id, "publication_id": publication_id},
            {"$set": update_data}
        )
        
        # If not found by publication_id, try by _id
        if result.modified_count == 0:
            try:
                result = database[PUBLICATIONS_COLLECTION].update_one(
                    {"supervisor_id": supervisor_id, "_id": ObjectId(publication_id)},
                    {"$set": update_data}
                )
            except Exception as e:
                print(f"⚠️  Could not convert publication_id to ObjectId: {e}")
        
        print(f"✏️ Update result: {result.modified_count} documents modified")
        
        # Recalculate scoreboard
        recalculate_scoreboard(supervisor_id)
        
        return {
            "status": "success",
            "message": "Publication updated",
            "modified_count": result.modified_count
        }
    except Exception as e:
        print(f"❌ Error updating publication: {e}")
        return {
            "status": "error",
            "message": f"Failed to update publication: {str(e)}"
        }


def delete_publication(supervisor_id: str, publication_id: str) -> dict:
    """Delete a publication and recalculate scoreboard"""
    try:
        database = get_db()
        if database is None:
            return {"status": "error", "message": "Database not connected"}
        
        # Try to delete by publication_id first
        result = database[PUBLICATIONS_COLLECTION].delete_one(
            {"supervisor_id": supervisor_id, "publication_id": publication_id}
        )
        
        # If not found, try by _id (for old documents)
        if result.deleted_count == 0:
            try:
                result = database[PUBLICATIONS_COLLECTION].delete_one(
                    {"supervisor_id": supervisor_id, "_id": ObjectId(publication_id)}
                )
            except Exception as e:
                print(f"⚠️  Could not convert publication_id to ObjectId: {e}")
        
        print(f"🗑️ Delete result: {result.deleted_count} documents deleted")
        
        # Recalculate scoreboard
        recalculate_scoreboard(supervisor_id)
        
        return {
            "status": "success",
            "message": "Publication deleted",
            "deleted_count": result.deleted_count
        }
    except Exception as e:
        print(f"❌ Error deleting publication: {e}")
        return {
            "status": "error",
            "message": f"Failed to delete publication: {str(e)}"
        }


def _build_preferred_embedding_text(title: str, abstract: str) -> str:
    """Build canonical text used for preferred project embeddings."""
    safe_title = (title or "").strip()
    safe_abstract = (abstract or "").strip()
    return f"Title: {safe_title}\nAbstract: {safe_abstract}".strip()


def _build_preferred_payload(title: str, abstract: str) -> dict:
    """Create preferred-project payload and regenerate embedding from current fields."""
    embedding_text = _build_preferred_embedding_text(title, abstract)
    payload = {
        "title": (title or "").strip(),
        "abstract": (abstract or "").strip(),
        "updated_at": datetime.utcnow()
    }

    embedding = generate_embedding(embedding_text)
    if embedding:
        payload["embedding"] = embedding

    return payload


def add_preferred_publication(supervisor_id: str, title: str, abstract: str) -> dict:
    """Insert a preferred publication and generate embedding for vector search."""
    try:
        database = get_db()
        if database is None:
            return {"status": "error", "message": "Database not connected"}

        payload = _build_preferred_payload(title, abstract)
        payload["supervisor_id"] = supervisor_id
        payload["created_at"] = datetime.utcnow()

        result = database[PREFERRED_PROJECTS_COLLECTION].insert_one(payload)
        publication_id = str(result.inserted_id)

        database[PREFERRED_PROJECTS_COLLECTION].update_one(
            {"_id": result.inserted_id},
            {"$set": {"publication_id": publication_id}}
        )

        return {
            "status": "success",
            "message": "Preferred publication added",
            "publication_id": publication_id
        }
    except Exception as e:
        print(f"❌ Error adding preferred publication: {e}")
        return {
            "status": "error",
            "message": f"Failed to add preferred publication: {str(e)}"
        }


def update_preferred_publication(
    supervisor_id: str,
    publication_id: str,
    title: str,
    abstract: str,
) -> dict:
    """Update a preferred publication and regenerate embedding from latest content."""
    try:
        database = get_db()
        if database is None:
            return {"status": "error", "message": "Database not connected"}

        update_data = _build_preferred_payload(title, abstract)

        result = database[PREFERRED_PROJECTS_COLLECTION].update_one(
            {"supervisor_id": supervisor_id, "publication_id": publication_id},
            {"$set": update_data}
        )

        if result.matched_count == 0:
            try:
                result = database[PREFERRED_PROJECTS_COLLECTION].update_one(
                    {"supervisor_id": supervisor_id, "_id": ObjectId(publication_id)},
                    {"$set": update_data}
                )
            except Exception as e:
                print(f"⚠️  Could not convert publication_id to ObjectId: {e}")

        if result.matched_count == 0:
            return {
                "status": "error",
                "message": "Preferred publication not found"
            }

        return {
            "status": "success",
            "message": "Preferred publication updated",
            "modified_count": result.modified_count
        }
    except Exception as e:
        print(f"❌ Error updating preferred publication: {e}")
        return {
            "status": "error",
            "message": f"Failed to update preferred publication: {str(e)}"
        }


def delete_preferred_publication(supervisor_id: str, publication_id: str) -> dict:
    """Delete a preferred publication by publication_id (or legacy ObjectId)."""
    try:
        database = get_db()
        if database is None:
            return {"status": "error", "message": "Database not connected"}

        result = database[PREFERRED_PROJECTS_COLLECTION].delete_one(
            {"supervisor_id": supervisor_id, "publication_id": publication_id}
        )

        if result.deleted_count == 0:
            try:
                result = database[PREFERRED_PROJECTS_COLLECTION].delete_one(
                    {"supervisor_id": supervisor_id, "_id": ObjectId(publication_id)}
                )
            except Exception as e:
                print(f"⚠️  Could not convert publication_id to ObjectId: {e}")

        if result.deleted_count == 0:
            return {
                "status": "error",
                "message": "Preferred publication not found"
            }

        return {
            "status": "success",
            "message": "Preferred publication deleted",
            "deleted_count": result.deleted_count
        }
    except Exception as e:
        print(f"❌ Error deleting preferred publication: {e}")
        return {
            "status": "error",
            "message": f"Failed to delete preferred publication: {str(e)}"
        }


def get_supervisor_publications(supervisor_id: str) -> dict:
    """
    Get all publications for a supervisor.
    
    Args:
        supervisor_id: Supervisor ID
    
    Returns:
        Dict with publications list
    """
    database = get_db()
    
    if database is None:
        return {"status": "error", "publications": [], "count": 0}
    
    try:
        publications = list(database[PUBLICATIONS_COLLECTION].find(
            {"supervisor_id": supervisor_id},
            {"_id": 0, "embedding": 0}  # Exclude embedding for speed
        ).sort("Year", -1))  # Sort by year descending
        publications = _serialize_docs(publications)
        
        return {
            "status": "success",
            "supervisor_id": supervisor_id,
            "publications": publications,
            "count": len(publications)
        }
    
    except Exception as e:
        print(f"⚠️  Failed to fetch publications: {e}")
        return {
            "status": "error",
            "publications": [],
            "count": 0,
            "message": str(e)
        }


def get_supervisor_profile(supervisor_id: str) -> dict:
    """
    Get complete supervisor profile (publications + scoreboard).
    Simple version - just return the data.
    """
    try:
        database = get_db()
        if database is None:
            return {"status": "error", "publications": [], "scoreboard": []}
        
        # Get publications
        pubs = list(database[PUBLICATIONS_COLLECTION].find(
            {"supervisor_id": supervisor_id},
            {"_id": 0, "embedding": 0}
        ).limit(100))
        
        # Get scoreboard
        scores = list(database[SCOREBOARD_COLLECTION].find(
            {"supervisor_id": supervisor_id}
        ).limit(100))

        # Get preferred publications (supervisor-added preferred_projects)
        preferred_pubs = list(database[PREFERRED_PROJECTS_COLLECTION].find(
            {"supervisor_id": supervisor_id},
            {"_id": 0, "embedding": 0}
        ).limit(100))
        
        # Serialize
        pubs = _serialize_docs(pubs)
        scores = _serialize_docs(scores)
        preferred_pubs = _serialize_docs(preferred_pubs)
        
        return {
            "status": "success",
            "supervisor_id": supervisor_id,
            "publications": pubs,
            "scoreboard": scores,
            "preferred_publications": preferred_pubs
        }
    except Exception as e:
        print(f"Error in get_supervisor_profile: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "publications": [], 
            "scoreboard": [],
            "error": str(e)
        }


# ============================================================================
# SUPERVISOR FULL STATUS MANAGEMENT
# ============================================================================

def get_supervisor_full_status(supervisor_id: str) -> dict:
    """
    Get the 'is_full' status for a supervisor.
    
    Args:
        supervisor_id: Supervisor ID (e.g., "SUP-001")
    
    Returns:
        Dict with is_full status
    """
    try:
        database = get_db()
        if database is None:
            return {
                "status": "success",
                "supervisor_id": supervisor_id,
                "is_full": False,
                "message": "Database not connected, defaulting to is_full=False"
            }
        
        # Find supervisor record
        supervisor = database[SUPERVISORS_COLLECTION].find_one(
            {"supervisor_id": supervisor_id}
        )
        
        if supervisor:
            return {
                "status": "success",
                "supervisor_id": supervisor_id,
                "is_full": supervisor.get("is_full", False)
            }
        else:
            # Supervisor doesn't exist yet, default to False
            return {
                "status": "success",
                "supervisor_id": supervisor_id,
                "is_full": False
            }
    
    except Exception as e:
        print(f"❌ Error getting supervisor full status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "is_full": False
        }


def toggle_supervisor_full_status(supervisor_id: str, is_full: bool) -> dict:
    """
    Toggle the 'is_full' status for a supervisor.
    Marks supervisor as full so they won't appear in recommendations.
    
    Args:
        supervisor_id: Supervisor ID (e.g., "SUP-001")
        is_full: Boolean indicating if supervisor is full
    
    Returns:
        Dict with updated status
    """
    try:
        database = get_db()
        if database is None:
            return {"status": "error", "message": "Database not connected"}
        
        # Upsert supervisor record
        result = database[SUPERVISORS_COLLECTION].update_one(
            {"supervisor_id": supervisor_id},
            {
                "$set": {
                    "supervisor_id": supervisor_id,
                    "is_full": is_full,
                    "updated_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        
        status_text = "full (will not appear in recommendations)" if is_full else "available"
        print(f"✅ Supervisor {supervisor_id} marked as {status_text}")
        
        return {
            "status": "success",
            "supervisor_id": supervisor_id,
            "is_full": is_full,
            "message": f"Supervisor marked as {status_text}"
        }
    
    except Exception as e:
        print(f"❌ Error updating supervisor full status: {e}")
        return {
            "status": "error",
            "message": f"Failed to update status: {str(e)}"
        }


def get_full_supervisors() -> list:
    """
    Get list of all supervisors marked as full.
    
    Returns:
        List of supervisor IDs that are marked as full
    """
    try:
        database = get_db()
        if database is None:
            return []
        
        full_supervisors = database[SUPERVISORS_COLLECTION].find(
            {"is_full": True}
        )
        
        return [doc["supervisor_id"] for doc in full_supervisors]
    
    except Exception as e:
        print(f"⚠️  Error fetching full supervisors: {e}")
        return []
