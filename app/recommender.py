from sentence_transformers import SentenceTransformer, util
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

model = SentenceTransformer("all-MiniLM-L6-v2")

PUBLICATIONS_COLLECTION = "publications"
PREFERRED_PROJECTS_COLLECTION = "preferred_projects"
PUBLICATIONS_VECTOR_INDEX = "vector_index"
PREFERRED_VECTOR_INDEX = "preferred_vector_index"
PREFERRED_SCORE_THRESHOLD = 0.80

# -------- ABBREVIATION MAP (FROZEN) --------
ABBR_MAP = {
    "AI": "Artificial Intelligence",
    "AGI": "Artificial General Intelligence",
    "CI": "Computational Intelligence",
    "ML": "Machine Learning",
    "DL": "Deep Learning",
    "CNN": "Convolutional Neural Networks",
    "NLP": "Natural Language Processing",
    "LLM": "Large Language Models",
    "XAI": "Explainable Artificial Intelligence",
    "HCI": "Human Computer Interaction",
    "UI": "User Interface",
    "UX": "User Experience",
    "UI/UX": "User Interface and User Experience",
    "Ui/Ux": "User Interface and User Experience",
    "XR": "Extended Reality",
    "TEL": "Technology Enhanced Learning",
    "ECG": "Electrocardiography"
}

def normalize_abbreviations(text: str) -> str:
    for abbr, full in ABBR_MAP.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text


@lru_cache(maxsize=32)
def _get_cached_domain_embeddings(domain_key):
    return model.encode(list(domain_key), convert_to_tensor=True)


def extract_proposal_domains(
    proposal_text,
    domain_list,
    top_k=3,
    similarity_threshold=0.38,
    proposal_embedding=None,
):
    if not domain_list:
        return []

    if proposal_embedding is None:
        proposal_text = normalize_abbreviations(proposal_text)
        proposal_embedding = model.encode(proposal_text, convert_to_tensor=True)

    domain_embs = _get_cached_domain_embeddings(tuple(domain_list))

    # Similarities
    similarities = util.cos_sim(proposal_embedding, domain_embs)[0].cpu().numpy()

    # 1️⃣ Always keep top-k domains
    top_indices = similarities.argsort()[::-1][:top_k]
    selected_indices = set(top_indices.tolist())

    # 2️⃣ Add domains above similarity threshold
    for idx, score in enumerate(similarities):
        if score >= similarity_threshold:
            selected_indices.add(idx)

    # Return final domain list
    return [domain_list[i] for i in selected_indices]


def _build_scoreboard_matches(scoreboard_df, extracted_domains):
    matches = {}

    for sup_id in scoreboard_df["supervisor_id"].unique():
        sup_rows = scoreboard_df[scoreboard_df["supervisor_id"] == sup_id]
        matched = sup_rows[sup_rows["domain"].isin(extracted_domains)]

        if matched.empty:
            continue

        total_score = float(matched["final_score"].sum())
        matches[sup_id] = {
            "supervisor_id": sup_id,
            "matched_domains": matched["domain"].tolist(),
            "scoreboard_score": total_score,
        }

    return matches


def _legacy_scoreboard_recommendations(scoreboard_df, extracted_domains):
    scoreboard_matches = _build_scoreboard_matches(scoreboard_df, extracted_domains)
    results = []

    for sup_id, match in scoreboard_matches.items():
        total_score = match["scoreboard_score"]
        reason = _build_recommendation_reason(
            matched_domains=match["matched_domains"],
            publication_title="",
            preferred_title="",
            mode="scoreboard_only",
            using_default_weights=True,
            has_global_preferred_matches=False,
            preferred_score=0.0,
        )
        results.append({
            "supervisor_id": sup_id,
            "matched_domains": match["matched_domains"],
            "score": round(total_score, 3),
            "reason": reason,
            "score_breakdown": {
                "publication_vector_component": 0.0,
                "preferred_vector_component": 0.0,
                "scoreboard_component": round(total_score, 3),
                "publication_vector_score": 0.0,
                "preferred_vector_score": 0.0,
                "scoreboard_score": round(total_score, 3),
                "mode": "scoreboard_only",
            }
        })

    if not results and not scoreboard_df.empty:
        top = scoreboard_df.sort_values("final_score", ascending=False).iloc[0]
        top_score = float(top["final_score"])
        reason = _build_recommendation_reason(
            matched_domains=[top["domain"]],
            publication_title="",
            preferred_title="",
            mode="scoreboard_only",
            using_default_weights=True,
            has_global_preferred_matches=False,
            preferred_score=0.0,
        )
        results.append({
            "supervisor_id": top["supervisor_id"],
            "matched_domains": [top["domain"]],
            "score": round(top_score, 3),
            "reason": reason,
            "score_breakdown": {
                "publication_vector_component": 0.0,
                "preferred_vector_component": 0.0,
                "scoreboard_component": round(top_score, 3),
                "publication_vector_score": 0.0,
                "preferred_vector_score": 0.0,
                "scoreboard_score": round(top_score, 3),
                "mode": "scoreboard_only",
            }
        })

    return sorted(results, key=lambda item: item["score"], reverse=True)


def _fetch_vector_matches(database, collection_name, index_name, query_embedding, title_field, limit=100):
    if database is None:
        return {}

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": max(limit * 2, 100),
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "supervisor_id": 1,
                "publication_id": 1,
                title_field: 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ]

    try:
        documents = list(database[collection_name].aggregate(pipeline))
    except Exception as exc:
        print(f"⚠️ Vector search failed for {collection_name}: {exc}")
        return {}

    matches = {}
    for document in documents:
        supervisor_id = document.get("supervisor_id")
        if not supervisor_id:
            continue

        score = float(document.get("score") or 0.0)
        current_best = matches.get(supervisor_id)
        if current_best is not None and current_best["score"] >= score:
            continue

        matches[supervisor_id] = {
            "score": score,
            "title": document.get(title_field, ""),
            "publication_id": document.get("publication_id"),
        }

    return matches


def _format_domains(domains, max_items=3):
    if not domains:
        return ""

    shown = domains[:max_items]
    suffix = f" and {len(domains) - max_items} more" if len(domains) > max_items else ""
    return ", ".join(shown) + suffix


def _build_recommendation_reason(
    matched_domains,
    publication_title,
    preferred_title,
    mode,
    using_default_weights,
    has_global_preferred_matches,
    preferred_score,
):
    sentences = []

    domains_text = _format_domains(matched_domains)
    if domains_text:
        sentences.append(f"This supervisor aligns with your proposal topics in {domains_text}.")

    if publication_title:
        sentences.append(f"A closely related publication was identified: \"{publication_title}\".")

    if preferred_title and preferred_score >= PREFERRED_SCORE_THRESHOLD:
        sentences.append(f"A strong preferred-project match was also found: \"{preferred_title}\".")

    if mode == "preferred_bonus":
        if using_default_weights:
            sentences.append(
                "The final rank combines publication similarity, historical domain suitability, and preferred-project alignment."
            )
        else:
            sentences.append(
                "The final rank uses your custom weighting across publication similarity, historical suitability, and preferred-project alignment."
            )
    elif mode == "fallback_no_preferred_match":
        if has_global_preferred_matches:
            sentences.append(
                "This score is based on publication similarity and historical domain suitability because this supervisor did not meet the preferred-project threshold."
            )
        elif using_default_weights:
            sentences.append(
                "No strong preferred-project matches were found for this search, so the score is based on publication similarity and historical domain suitability."
            )
        else:
            sentences.append(
                "Using your custom fallback weighting, the score is based on publication similarity and historical domain suitability."
            )
    else:
        sentences.append(
            "This recommendation is based on domain suitability from the historical scoreboard."
        )

    return " ".join(sentences)


def _read_weight_group(config, key):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _read_weight_value(group, key, default_value):
    if group is None:
        return default_value
    if isinstance(group, dict):
        return float(group.get(key, default_value))
    return float(getattr(group, key, default_value))


def _resolve_scoring_weights(scoring_config=None):
    default_preferred = {
        "publication": 30.0,
        "scoreboard": 30.0,
        "preferred": 40.0,
    }
    default_fallback = {
        "publication": 60.0,
        "scoreboard": 40.0,
    }

    if scoring_config is None:
        preferred = default_preferred.copy()
        fallback = default_fallback.copy()
        return preferred, fallback, True

    preferred_group = _read_weight_group(scoring_config, "preferred_bonus")
    fallback_group = _read_weight_group(scoring_config, "fallback")

    preferred = {
        "publication": _read_weight_value(preferred_group, "publication", default_preferred["publication"]),
        "scoreboard": _read_weight_value(preferred_group, "scoreboard", default_preferred["scoreboard"]),
        "preferred": _read_weight_value(preferred_group, "preferred", default_preferred["preferred"]),
    }
    fallback = {
        "publication": _read_weight_value(fallback_group, "publication", default_fallback["publication"]),
        "scoreboard": _read_weight_value(fallback_group, "scoreboard", default_fallback["scoreboard"]),
    }

    for name, value in {**preferred, **fallback}.items():
        if value < 0:
            raise ValueError(f"Invalid scoring_config: percentage '{name}' must be >= 0")

    preferred_total = preferred["publication"] + preferred["scoreboard"] + preferred["preferred"]
    fallback_total = fallback["publication"] + fallback["scoreboard"]

    if abs(preferred_total - 100.0) > 0.01:
        raise ValueError("Invalid scoring_config: preferred_bonus percentages must sum to 100")
    if abs(fallback_total - 100.0) > 0.01:
        raise ValueError("Invalid scoring_config: fallback percentages must sum to 100")

    return preferred, fallback, False


def recommend_supervisors(proposal_text, scoreboard_df, database=None, scoring_config=None):
    if scoreboard_df is None or scoreboard_df.empty:
        return []

    preferred_weights_pct, fallback_weights_pct, using_default_weights = _resolve_scoring_weights(scoring_config)
    preferred_weights = {k: v / 100.0 for k, v in preferred_weights_pct.items()}
    fallback_weights = {k: v / 100.0 for k, v in fallback_weights_pct.items()}

    normalized_proposal = normalize_abbreviations(proposal_text)
    proposal_embedding = model.encode(normalized_proposal, convert_to_tensor=True)

    domains = scoreboard_df["domain"].dropna().unique().tolist()
    extracted_domains = extract_proposal_domains(
        proposal_text,
        domains,
        proposal_embedding=proposal_embedding,
    )
    scoreboard_matches = _build_scoreboard_matches(scoreboard_df, extracted_domains)

    query_embedding = proposal_embedding.detach().cpu().tolist()

    with ThreadPoolExecutor(max_workers=2) as executor:
        publications_future = executor.submit(
            _fetch_vector_matches,
            database,
            PUBLICATIONS_COLLECTION,
            PUBLICATIONS_VECTOR_INDEX,
            query_embedding,
            "Title",
        )
        preferred_future = executor.submit(
            _fetch_vector_matches,
            database,
            PREFERRED_PROJECTS_COLLECTION,
            PREFERRED_VECTOR_INDEX,
            query_embedding,
            "title",
        )
        publication_matches = publications_future.result()
        preferred_matches = preferred_future.result()

    if not publication_matches and not preferred_matches:
        return _legacy_scoreboard_recommendations(scoreboard_df, extracted_domains)

    qualifying_preferred = {
        sup_id: match
        for sup_id, match in preferred_matches.items()
        if match["score"] >= PREFERRED_SCORE_THRESHOLD
    }
    use_preferred_weighting = bool(qualifying_preferred)

    candidate_supervisors = set(scoreboard_matches) | set(publication_matches)
    if use_preferred_weighting:
        candidate_supervisors |= set(qualifying_preferred)

    full_supervisor_ids = set()
    if database is not None:
        try:
            supervisors_collection = database.get_collection("supervisors")
            # Fetch ALL full supervisors in ONE query (avoid N+1 problem)
            full_docs = supervisors_collection.find({"is_full": True})
            full_supervisor_ids = {doc["supervisor_id"] for doc in full_docs}
            if full_supervisor_ids:
                print(f"📋 Found {len(full_supervisor_ids)} full supervisors to filter out")
        except Exception as e:
            print(f"⚠️  Warning: Could not check full supervisors: {e}")
            # Continue with original candidate list if checking fails

    results = []

    for supervisor_id in candidate_supervisors:
        scoreboard_match = scoreboard_matches.get(supervisor_id, {})
        publication_match = publication_matches.get(supervisor_id, {})
        preferred_match = qualifying_preferred.get(supervisor_id, {}) if use_preferred_weighting else {}

        scoreboard_score = float(scoreboard_match.get("scoreboard_score", 0.0))
        publication_score = float(publication_match.get("score", 0.0))
        preferred_score = float(preferred_match.get("score", 0.0))

        if use_preferred_weighting:
            publication_component = preferred_weights["publication"] * publication_score
            scoreboard_component = preferred_weights["scoreboard"] * scoreboard_score
            preferred_component = preferred_weights["preferred"] * preferred_score
            mode = "preferred_bonus"
        else:
            publication_component = fallback_weights["publication"] * publication_score
            scoreboard_component = fallback_weights["scoreboard"] * scoreboard_score
            preferred_component = 0.0
            mode = "fallback_no_preferred_match"

        final_score = publication_component + scoreboard_component + preferred_component
        if final_score <= 0:
            continue

        matched_domains = scoreboard_match.get("matched_domains", [])
        publication_title = publication_match.get("title", "")
        preferred_title = preferred_match.get("title", "")

        score_breakdown = {
            "publication_vector_component": round(publication_component, 3),
            "preferred_vector_component": round(preferred_component, 3),
            "scoreboard_component": round(scoreboard_component, 3),
            "publication_vector_score": round(publication_score, 3),
            "preferred_vector_score": round(preferred_score, 3),
            "scoreboard_score": round(scoreboard_score, 3),
            "mode": mode,
        }

        if not using_default_weights:
            score_breakdown["weights_used"] = {
                "preferred_bonus": {
                    "publication": round(preferred_weights_pct["publication"], 3),
                    "scoreboard": round(preferred_weights_pct["scoreboard"], 3),
                    "preferred": round(preferred_weights_pct["preferred"], 3),
                },
                "fallback": {
                    "publication": round(fallback_weights_pct["publication"], 3),
                    "scoreboard": round(fallback_weights_pct["scoreboard"], 3),
                },
                "is_default": using_default_weights,
            }

        reason = _build_recommendation_reason(
            matched_domains=matched_domains,
            publication_title=publication_title,
            preferred_title=preferred_title,
            mode=mode,
            using_default_weights=using_default_weights,
            has_global_preferred_matches=use_preferred_weighting,
            preferred_score=preferred_score,
        )

        results.append({
            "supervisor_id": supervisor_id,
            "matched_domains": matched_domains,
            "score": round(final_score, 3),
            "reason": reason,
            "publication_match_title": publication_title,
            "preferred_match_title": preferred_title,
            "preferred_publication_id": preferred_match.get("publication_id"),
            "score_breakdown": score_breakdown
        })

    if not results:
        return _legacy_scoreboard_recommendations(scoreboard_df, extracted_domains)

    # Filter out full supervisors from results to ensure only available ones are returned
    available_results = [r for r in results if r['supervisor_id'] not in full_supervisor_ids]
    
    # If all results were filtered out (all were full), return empty
    if not available_results:
        print("All candidate supervisors are marked as full. Returning empty recommendations.")
        return []
    
    # Sort by score and return (takes top 5 available in main.py)
    return sorted(available_results, key=lambda item: item["score"], reverse=True)
