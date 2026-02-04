from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

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


def extract_proposal_domains(
    proposal_text,
    domain_list,
    top_k=3,
    similarity_threshold=0.38
):
    # Normalize abbreviations
    proposal_text = normalize_abbreviations(proposal_text)

    # Encode
    proposal_emb = model.encode(proposal_text, convert_to_tensor=True)
    domain_embs = model.encode(domain_list, convert_to_tensor=True)

    # Similarities
    similarities = util.cos_sim(proposal_emb, domain_embs)[0].cpu().numpy()

    # 1️⃣ Always keep top-k domains
    top_indices = similarities.argsort()[::-1][:top_k]
    selected_indices = set(top_indices.tolist())

    # 2️⃣ Add domains above similarity threshold
    for idx, score in enumerate(similarities):
        if score >= similarity_threshold:
            selected_indices.add(idx)

    # Return final domain list
    return [domain_list[i] for i in selected_indices]


def recommend_supervisors(proposal_text, scoreboard_df):
    domains = scoreboard_df["domain"].unique().tolist()
    extracted_domains = extract_proposal_domains(proposal_text, domains)

    results = []

    for sup_id in scoreboard_df["supervisor_id"].unique():
        sup_rows = scoreboard_df[scoreboard_df["supervisor_id"] == sup_id]

        matched = sup_rows[sup_rows["domain"].isin(extracted_domains)]
        if matched.empty:
            continue

        total_score = matched["final_score"].sum()

        results.append({
            "supervisor_id": sup_id,
            "matched_domains": matched["domain"].tolist(),
            "score": round(float(total_score), 3),
            "reason": (
                f"Matched proposal domains {matched['domain'].tolist()} "
                f"using semantic similarity and suitability scores"
            )
        })

    # Fallback logic (unchanged)
    if not results:
        top = scoreboard_df.sort_values("final_score", ascending=False).iloc[0]
        results.append({
            "supervisor_id": top["supervisor_id"],
            "matched_domains": [top["domain"]],
            "score": round(float(top["final_score"]), 3),
            "reason": "Best match in primary proposal domain"
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)
