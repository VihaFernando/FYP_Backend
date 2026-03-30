"""
Automated Publication Pipeline - Full Notebook Logic Ported to Production

Implements all steps from FYP_Implementation.ipynb:
1. Text preprocessing (lowercase, remove symbols, lemmatization)
2. TF-IDF domain extraction
3. Supervisor-domain aggregation
4. Experience score (T_EXP = 5)
5. Recency score (current year = 2026)
6. Preference score (declared_interests match)
7. Saturation penalty (T_SAT = 5)
8. Final score formula (locked weights)
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Ensure NLTK data is downloaded
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

CURRENT_YEAR = 2026
STOP_WORDS = set(stopwords.words("english"))

# Constants from notebook
T_EXP = 5  # Experience threshold
T_SAT = 5  # Saturation threshold

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


# ============================================================================
# STEP 1: TEXT PREPROCESSING (From Notebook Section 7-8)
# ============================================================================

def preprocess_text(text: str) -> str:
    """
    Text preprocessing pipeline:
    - Lowercase
    - Remove symbols and numbers
    - Remove stopwords
    - Lemmatize tokens
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove symbols and numbers
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # spaCy processing
    if nlp is None:
        return text
    
    doc = nlp(text)
    
    # Lemmatization + stopword removal
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOP_WORDS
        and token.lemma_ != "-pron-"
        and len(token.lemma_) > 2
    ]
    
    return " ".join(tokens)


# ============================================================================
# STEP 2: DOMAIN EXTRACTION WITH TF-IDF VALIDATION (From Notebook Section 3-5)
# ============================================================================

def parse_domains(domain_str: str) -> list:
    """Parse comma-separated domain list"""
    if pd.isna(domain_str):
        return []
    return [d.strip() for d in str(domain_str).split(",") if d.strip()]


def extract_domains_with_tfidf(publications_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and validate domains using TF-IDF.
    Only use data that's provided; don't add fictional TF-IDF scoring.
    """
    # If combined_text exists, use TF-IDF; otherwise use domain field directly
    if "combined_text" not in publications_df.columns:
        # No preprocessing done yet - return domains as-is
        publications_df["extracted_domains"] = publications_df.get(
            "domain of the publication", 
            publications_df.get("domain", "")
        )
        return publications_df
    
    # Build TF-IDF model on combined_text
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, min_df=1)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(publications_df["combined_text"].fillna(""))
        feature_names = np.array(vectorizer.get_feature_names_out())
    except:
        publications_df["extracted_domains"] = publications_df.get(
            "domain of the publication", 
            publications_df.get("domain", "")
        )
        return publications_df
    
    # Score domains
    def domain_tfidf_score(domain: str, row_vector) -> float:
        tokens = domain.lower().split()
        score = 0.0
        for n in range(1, min(3, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i+n])
                if phrase in feature_names:
                    idx = np.where(feature_names == phrase)[0]
                    if len(idx) > 0:
                        score += row_vector[idx[0]]
        return score
    
    def extract_valid_domains(row):
        domain_candidates = parse_domains(
            row.get("domain of the publication") or row.get("domain", "")
        )
        row_idx = row.name
        row_vector = tfidf_matrix[row_idx].toarray()[0] if row_idx < tfidf_matrix.shape[0] else np.array([])
        
        scored_domains = []
        for domain in domain_candidates:
            score = domain_tfidf_score(domain, row_vector)
            if score > 0:
                scored_domains.append((domain, score))
        
        scored_domains.sort(key=lambda x: x[1], reverse=True)
        return " | ".join([d[0] for d in scored_domains[:5]])
    
    publications_df["extracted_domains"] = publications_df.apply(extract_valid_domains, axis=1)
    return publications_df


# ============================================================================
# STEP 3: SUPERVISOR-DOMAIN AGGREGATION (From Notebook Section 3)
# ============================================================================

def aggregate_supervisor_domains(publications_df: pd.DataFrame, existing_scoreboard: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate publications by supervisor + domain.
    If existing_scoreboard provided, merge and update.
    """
    
    # Ensure columns exist
    publications_df["extracted_domains"] = publications_df.get("extracted_domains", "").astype(str)
    publications_df["supervisor_id"] = publications_df.get("supervisor_id", "").astype(str)
    publications_df["year"] = pd.to_numeric(publications_df.get("year", publications_df.get("latest_year", 2026)), errors="coerce")
    
    # Explode domains
    publications_df["domain"] = publications_df["extracted_domains"].str.split(r"\s*\|\s*")
    df_exploded = publications_df.explode("domain")
    df_exploded["domain"] = df_exploded["domain"].str.strip()
    df_exploded = df_exploded[df_exploded["domain"] != ""]
    
    # Aggregate
    agg = (
        df_exploded
        .groupby(["supervisor_id", "domain"])
        .agg(
            domain_count=("domain", "count"),
            latest_year=("year", "max")
        )
        .reset_index()
    )
    
    # Compute totals
    total_counts = (
        agg
        .groupby("supervisor_id")["domain_count"]
        .sum()
        .reset_index(name="total_domain_count")
    )
    agg = agg.merge(total_counts, on="supervisor_id", how="left")
    
    # Dominance
    agg["dominance"] = agg["domain_count"] / agg["total_domain_count"]
    
    return agg[["supervisor_id", "domain", "domain_count", "dominance", "latest_year"]]


# ============================================================================
# STEP 4: EXPERIENCE SCORE (From Notebook)
# ============================================================================

def compute_experience_score(scoreboard: pd.DataFrame) -> pd.DataFrame:
    """Experience score: domain_count / T_EXP (capped at 1.0)"""
    scoreboard["experience_score"] = (scoreboard["domain_count"] / T_EXP).clip(upper=1.0)
    return scoreboard


# ============================================================================
# STEP 5: RECENCY SCORE (From Notebook)
# ============================================================================

def compute_recency_score(scoreboard: pd.DataFrame) -> pd.DataFrame:
    """
    Recency scoring:
    - 0-5 years: 1.0
    - 6-10 years: 0.7
    - 11+ years: 0.4
    """
    def recency(year):
        if pd.isna(year):
            return 0.4
        age = CURRENT_YEAR - int(year)
        if age <= 5:
            return 1.0
        elif age <= 10:
            return 0.7
        else:
            return 0.4
    
    scoreboard["recency_score"] = scoreboard["latest_year"].apply(recency)
    return scoreboard


# ============================================================================
# STEP 6: PREFERENCE SCORE (From Notebook)
# ============================================================================

def compute_preference_score(
    scoreboard: pd.DataFrame, 
    publications_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Preference score: 1.0 if domain matches supervisor's declared_interests, else 0.0
    """
    
    def normalize(text):
        return str(text).strip().lower()
    
    # Build supervisor → declared interests map
    publications_df["declared_interests"] = publications_df.get("declared_interests", "").astype(str)
    
    sup_declared = {}
    for sup_id, group in publications_df.groupby("supervisor_id"):
        interests = set()
        for text in group["declared_interests"]:
            tokens = re.split(r"[;,|/]", text)
            for t in tokens:
                t = t.strip()
                if t in ABBR_MAP:
                    interests.add(normalize(ABBR_MAP[t]))
                elif len(t) > 3:
                    interests.add(normalize(t))
        sup_declared[sup_id] = interests
    
    # Add missing declared domains as rows
    new_rows = []
    for sup_id, interests in sup_declared.items():
        existing_domains = set(
            scoreboard.loc[scoreboard["supervisor_id"] == sup_id, "domain"]
            .astype(str)
            .apply(normalize)
        )
        
        for interest in interests:
            if interest not in existing_domains:
                new_rows.append({
                    "supervisor_id": sup_id,
                    "domain": interest.title(),
                    "domain_count": 0,
                    "dominance": 0.0,
                    "latest_year": CURRENT_YEAR,
                    "experience_score": 0.0,
                    "recency_score": 0.0
                })
    
    if new_rows:
        scoreboard = pd.concat([scoreboard, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Compute preference score
    def compute_pref(row):
        sup_id = row["supervisor_id"]
        domain_norm = normalize(row["domain"])
        declared = sup_declared.get(sup_id, set())
        return 1.0 if domain_norm in declared else 0.0
    
    scoreboard["preference_score"] = scoreboard.apply(compute_pref, axis=1)
    return scoreboard


# ============================================================================
# STEP 7: SATURATION PENALTY (From Notebook)
# ============================================================================

def compute_saturation_penalty(scoreboard: pd.DataFrame) -> pd.DataFrame:
    """
    Saturation penalty: log(domain_count / T_SAT) if domain_count > T_SAT, else 0.0
    """
    def penalty(count):
        if count <= T_SAT:
            return 0.0
        else:
            return np.log(count / T_SAT)
    
    scoreboard["saturation_penalty"] = scoreboard["domain_count"].apply(penalty)
    return scoreboard


# ============================================================================
# STEP 8: FINAL SCORE (From Notebook)
# ============================================================================

def compute_final_score(scoreboard: pd.DataFrame) -> pd.DataFrame:
    """
    Final score formula (locked weights):
    0.40 * experience_score
    + 0.25 * recency_score
    + 0.20 * preference_score
    - 0.15 * saturation_penalty
    """
    scoreboard["final_score"] = (
        0.40 * scoreboard["experience_score"]
        + 0.25 * scoreboard["recency_score"]
        + 0.20 * scoreboard["preference_score"]
        - 0.15 * scoreboard["saturation_penalty"]
    ).clip(lower=0.0)
    
    return scoreboard


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

def process_new_publications(
    publications_data: list,
    existing_scoreboard_path: str = "data/final_scoreboard.csv",
    skip_csv_merge: bool = True
) -> pd.DataFrame:
    """
    Full pipeline: takes raw publication data and returns updated scoreboard.
    
    Args:
        publications_data: List of dicts with keys: supervisor_id, title, abstract, 
                          domain of the publication, year, declared_interests
        existing_scoreboard_path: Path to existing scoreboard CSV (optional)
        skip_csv_merge: If True, don't load/merge CSV (default True for MongoDB workflow)
    
    Returns:
        Updated scoreboard DataFrame
    """
    
    # 1. Convert to DataFrame
    publications_df = pd.DataFrame(publications_data)
    
    # 2. Preprocess text
    publications_df["combined_raw_text"] = (
        publications_df.get("title", "").astype(str) + " " +
        publications_df.get("abstract", "").astype(str) + " " +
        publications_df.get("domain of the publication", "").astype(str)
    )
    publications_df["combined_text"] = publications_df["combined_raw_text"].apply(preprocess_text)
    
    # 3. Extract domains
    publications_df = extract_domains_with_tfidf(publications_df)
    
    # 4. Aggregate supervisor-domains
    new_scoreboard = aggregate_supervisor_domains(publications_df)
    
    # 5. Skip CSV merge by default for MongoDB workflow
    if skip_csv_merge:
        scoreboard = new_scoreboard.copy()
    else:
        # Load existing scoreboard if available (CSV workflow only)
        try:
            existing_scoreboard = pd.read_csv(existing_scoreboard_path)
            # Merge: keep all columns from existing, update with new values
            scoreboard = pd.concat([existing_scoreboard, new_scoreboard], ignore_index=True)
            # Re-aggregate to combine duplicate supervisor-domain pairs
            scoreboard = (
                scoreboard
                .groupby(["supervisor_id", "domain"])
                .agg({
                    "domain_count": "sum",
                    "dominance": "first",
                    "latest_year": "max"
                })
                .reset_index()
            )
            # Recompute total counts
            total_counts = (
                scoreboard
                .groupby("supervisor_id")["domain_count"]
                .sum()
                .reset_index(name="total_domain_count")
            )
            scoreboard = scoreboard.merge(total_counts, on="supervisor_id", how="left")
            scoreboard["dominance"] = scoreboard["domain_count"] / scoreboard["total_domain_count"]
        except FileNotFoundError:
            scoreboard = new_scoreboard.copy()
    
    # 6-8. Apply scoring functions
    scoreboard = compute_experience_score(scoreboard)
    scoreboard = compute_recency_score(scoreboard)
    scoreboard = compute_preference_score(scoreboard, publications_df)
    scoreboard = compute_saturation_penalty(scoreboard)
    scoreboard = compute_final_score(scoreboard)
    
    # Clean up temporary columns
    scoreboard = scoreboard[[
        "supervisor_id", "domain", "domain_count", "dominance", "latest_year",
        "experience_score", "recency_score", "preference_score", 
        "saturation_penalty", "final_score"
    ]]
    
    return scoreboard.reset_index(drop=True)


# ============================================================================
# UTILITY FUNCTION: UPDATE SCOREBOARD FILE
# ============================================================================

def update_scoreboard_file(scoreboard_df: pd.DataFrame, output_path: str = "data/final_scoreboard.csv"):
    """Save updated scoreboard to CSV"""
    scoreboard_df.to_csv(output_path, index=False)
    print(f"✅ Scoreboard updated: {output_path}")
    return scoreboard_df
