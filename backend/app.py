from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
import pandas as pd
import csv
from io import StringIO
import numpy as np
import traceback
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
import os
import json
import re
import requests
import copy
from typing import Any, Dict, Tuple
import time as time_module
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Feedback Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# In-memory store (MVP). Resets when server restarts.
DEMO_STATE = None
RATE = {}
STATE = {
    "df": None,
    "vectorizer": None,
    "nn_model": None,
    "tfidf_matrix": None,
    "kmeans": None,
    "theme_labels": None,
    "insight_cache": {},
    "rate_limit": {"ts": 0.0, "retry_after": 0}
}

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")


REQUIRED_COLS = ["id", "text", "source", "created_on", "segment", "product_area"]

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names (trim spaces)
    df.columns = [c.strip() for c in df.columns]

    # Keep only required cols (and ensure they exist)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()

    # Basic cleaning
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])

    # Parse date (keep invalid as NaT for now)
    df["created_on"] = pd.to_datetime(df["created_on"], errors="coerce")

    return df

def rate_limit(ip: str, limit=5, window=60):
    now = time_module.time()
    RATE.setdefault(ip, [])
    RATE[ip] = [t for t in RATE[ip] if now - t < window]
    if len(RATE[ip]) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    RATE[ip].append(now)

    
def build_state_from_df(df_clean: pd.DataFrame, k: int = 10) -> dict:
    texts = df_clean["text"].tolist()

    vectorizer, tfidf_matrix, nn_model = build_embeddings_and_index(texts)
    km, theme_ids = run_kmeans(tfidf_matrix, k)

    df2 = df_clean.copy()
    df2["theme_id"] = theme_ids.astype(int)

    # lightweight sentiment
    df2["sentiment_label"] = "NEUTRAL"
    df2["sentiment_score"] = 0.0

    theme_labels = label_themes_tfidf(df2["text"].tolist(), theme_ids, topn=5)

    return {
        "df": df2,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "nn_model": nn_model,
        "kmeans": km,
        "theme_labels": theme_labels,
        "insight_cache": {},
    }


def build_embeddings_and_index(texts: list[str]):
    """
    Lightweight indexing for deployment:
    - TF-IDF vectors (sparse)
    - NearestNeighbors over cosine distance
    Returns: (vectorizer, tfidf_matrix, nn_model)
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_model.fit(tfidf_matrix)

    return vectorizer, tfidf_matrix, nn_model
    

def run_kmeans(X, k: int):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=256)
    theme_ids = km.fit_predict(X)
    return km, theme_ids
    

def label_themes_tfidf(texts: list[str], theme_ids: np.ndarray, topn: int = 5) -> dict[int, str]:
    # Fit TF-IDF over all texts
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())

    labels = {}
    for t in sorted(set(theme_ids)):
        idx = np.where(theme_ids == t)[0]
        if len(idx) == 0:
            labels[int(t)] = "empty"
            continue
        # Mean tf-idf scores across docs in theme
        mean_scores = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:topn]
        keywords = [terms[i] for i in top_idx if mean_scores[i] > 0]
        labels[int(t)] = ", ".join(keywords[:topn]) if keywords else f"theme_{int(t)}"
    return labels

def compute_trend(df):
    df = df.dropna(subset=["created_on"]).copy()
    df["week"] = df["created_on"].dt.to_period("W")

    trends = {}
    for theme_id, g in df.groupby("theme_id"):
        counts = g.groupby("week").size().sort_index()
        if len(counts) < 2:
            trends[int(theme_id)] = 0.0
            continue

        recent = counts.iloc[-1]
        previous = counts.iloc[-2]
        trends[int(theme_id)] = (recent - previous) / max(previous, 1)

    return trends

def score_themes(df):
    trends = compute_trend(df)

    theme_scores = {}
    for theme_id, g in df.groupby("theme_id"):
        volume = len(g)
        negativity = abs(g[g["sentiment_score"] < 0]["sentiment_score"].mean()) if not g[g["sentiment_score"] < 0].empty else 0
        trend = trends.get(int(theme_id), 0.0)

        priority = (
            0.40 * volume +
            0.35 * negativity +
            0.25 * trend
        )

        if volume < 10:
            confidence = "low"
        elif volume < 20:
            confidence = "medium"
        else:
            confidence = "high"


        theme_scores[int(theme_id)] = {
            "volume": volume,
            "negativity": round(float(negativity), 3),
            "trend": round(float(trend), 3),
            "priority_score": round(float(priority), 3),
            "confidence": confidence
        }
        
    if theme_scores:
       max_priority = max(v["priority_score"] for v in theme_scores.values())
       for tid, v in theme_scores.items():
            raw = v["priority_score"]
            v["priority_score_100"] = round((raw / max_priority) * 100, 1) if max_priority > 0 else 0.0

    return theme_scores

def select_evidence(df, theme_id, n=10):
    subset = df[df["theme_id"] == theme_id]
    return subset.head(n)[
        ["id", "text", "source", "segment", "product_area", "created_on"]
    ].to_dict(orient="records")

def _extract_json_object(text: str) -> dict:
    if not text:
        raise ValueError("Empty model output")

    # Remove markdown fences if present
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        t = max(parts, key=len).strip()
        t = re.sub(r"^json\s*", "", t, flags=re.IGNORECASE).strip()

    # Extract outermost JSON object
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found. Got: {t[:200]}")

    obj = t[start:end+1]
    return json.loads(obj)


def _looks_like_prompt_echo(text: str) -> bool:
    t = (text or "").lower()
    return ("evidence:" in t and "return json" in t) or ("you are assisting a product manager" in t)


def build_insight_prompt(evidence_texts: list[str]) -> str:
    numbered = "\n".join([f"{i}. {t}" for i, t in enumerate(evidence_texts)])
    return f"""
Generate a product insight from evidence.

Output MUST be valid JSON ONLY. No markdown. No extra text.

EVIDENCE:
{numbered}

Return JSON with EXACT keys:
{{
  "title": "8-12 words",
  "summary": "2-3 sentences grounded in evidence",
  "recommended_action": "1-3 actions supported by evidence",
  "success_metric": "one measurable metric",
  "risks": "2-3 realistic risks",
  "evidence_refs": [0,1]
}}
""".strip()

def generate_insight_from_evidence_openrouter(evidence_texts: list[str]) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    prompt = build_insight_prompt(evidence_texts)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # These two are recommended by OpenRouter; not required, but helps routing.
        "HTTP-Referer": "https://feedback-insightful.lovable.app",
        "X-Title": "Feedback Insight Dashboard",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"].get("content", "").strip()

    # Parse JSON object from content
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON found in model output: {content[:200]}")

    return json.loads(content[start:end+1])


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a CSV, validate required columns, clean it, and store in memory.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()

    # Handle common encodings robustly
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    # Read CSV
    try:
        df = pd.read_csv(
    StringIO(text),
    sep=";",
    engine="python",
    quotechar='"',
    escapechar="\\",
)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {str(e)}")

    raw_rows = len(df)
  
    # Validate + clean
    df_clean = clean_df(df)

    # Store in memory
    STATE["df"] = df_clean
    texts = df_clean["text"].tolist()
    vectorizer, tfidf_matrix, nn_model = build_embeddings_and_index(texts)
    
    STATE["vectorizer"] = vectorizer
    STATE["tfidf_matrix"] = tfidf_matrix
    STATE["nn_model"] = nn_model

    
    # --- Clustering (Jan 6) ---
    k = 10
    km, theme_ids = run_kmeans(tfidf_matrix, k)
    STATE["kmeans"] = km
    
    STATE["df"] = STATE["df"].copy()
    STATE["df"]["theme_id"] = theme_ids.astype(int)
    texts_all = STATE["df"]["text"].tolist()
    STATE["theme_labels"] = label_themes_tfidf(texts_all, theme_ids, topn=5)

    
    # Lightweight sentiment for deployment (no transformers)
    STATE["df"]["sentiment_label"] = "NEUTRAL"
    STATE["df"]["sentiment_score"] = 0.0


    return {
        "message": "Upload successful",
        "rows_loaded": int(len(df_clean)),
        "raw_rows_read": int(raw_rows),
        "columns": list(df_clean.columns),
        "embeddings_built": True,
        "themes_built": True,
        "num_themes": int(k)
}

@app.get("/preview")
def preview(n: int = 5):
    if STATE["df"] is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet")
    return STATE["df"].head(n).to_dict(orient="records")


@app.get("/similar")
def search_similar(query: str, k: int = 5):
    if STATE.get("vectorizer") is None or STATE.get("nn_model") is None:
        return []

    q_vec = STATE["vectorizer"].transform([query])
    dists, idxs = STATE["nn_model"].kneighbors(q_vec, n_neighbors=k)

    results = []
    for dist, idx in zip(dists[0], idxs[0]):
        score = 1.0 - float(dist)  # cosine distance -> similarity-ish
        results.append((int(idx), score))
    return {
    "query": query,
    "k": k,
    "results": [{"row_index": int(idx), "score": float(score)} for idx, score in results]
}


@app.get("/themes")
def themes():
    if STATE["df"] is None or "theme_id" not in STATE["df"].columns:
        raise HTTPException(status_code=400, detail="No themes available. Upload data first.")

    df = STATE["df"]
    labels = STATE["theme_labels"] or {}

    scores = score_themes(df)
    
    out = []
    for theme_id, g in df.groupby("theme_id"):
        
        metrics = scores.get(int(theme_id), {})
        out.append({
            "theme_id": int(theme_id),
            "label": labels.get(int(theme_id), f"theme_{int(theme_id)}"),
            "count": int(len(g)),
            "priority_score": metrics.get("priority_score_100", 0.0),
            "confidence": metrics.get("confidence"),
            "score_components": {
                "volume": metrics.get("volume"),
                "negativity": metrics.get("negativity"),
                "trend": metrics.get("trend"),
            },
            "top_product_areas": g["product_area"].value_counts().head(3).to_dict(),
            "top_sources": g["source"].value_counts().head(3).to_dict(),
        })

    # Sort by size (later you'll sort by priority score)
    out.sort(key=lambda x: x["priority_score"], reverse=True)
    return {"num_themes": int(df["theme_id"].nunique()), "themes": out}

@app.get("/themes/{theme_id}")
def theme_detail(theme_id: int, n: int = 10):
    if STATE["df"] is None or STATE["theme_labels"] is None:
        raise HTTPException(status_code=400, detail="No themes available. Upload data first.")

    df = STATE["df"]
    label = (STATE.get("theme_labels") or {}).get(int(theme_id), f"theme_{int(theme_id)}")

    subset = df[df["theme_id"] == theme_id].copy()
    if subset.empty:
        raise HTTPException(status_code=404, detail="Theme not found")

    # compute scores and pull this theme's metrics
    scores = score_themes(df)
    metrics = scores.get(int(theme_id), {})
    priority_score = metrics.get("priority_score", 0.0)
    confidence = metrics.get("confidence", "low")

    # breakdowns (keep your existing logic; placeholders below)
    product_area_breakdown = subset["product_area"].value_counts().to_dict()
    source_breakdown = subset["source"].value_counts().to_dict()
    segment_breakdown = subset["segment"].value_counts().to_dict()
    
    # Representative examples (simple MVP): first n rows
    examples = subset.head(n)[["id", "text", "source", "segment", "product_area", "created_on"]].to_dict(orient="records")

    return {
        "theme_id": int(theme_id),
        "label": label,
        "count": int(len(subset)),
        "priority_score": metrics.get("priority_score_100", 0.0),
        "confidence": confidence,
        "score_components": {
            "volume": int(metrics.get("volume", len(subset))),
            "negativity": float(metrics.get("negativity", 0.0)),
            "trend": float(metrics.get("trend", 0.0)),
        },
        "product_area_breakdown": product_area_breakdown,
        "source_breakdown": source_breakdown,
        "segment_breakdown": segment_breakdown,
        "examples": examples,
    }

INSIGHT_TTL_SECONDS = 60 * 30  # 30 minutes

@app.get("/themes/{theme_id}/insight")
def theme_insight(theme_id: int, n: int = 10, force: bool = False):
    if STATE["df"] is None:
        raise HTTPException(status_code=400, detail="No data available. Load demo first.")

    df = STATE["df"]
    subset = df[df["theme_id"] == theme_id].copy()
    if subset.empty:
        raise HTTPException(status_code=404, detail="Theme not found")

    # Evidence list
    evidence_texts = subset.head(min(n, len(subset)))["text"].tolist()

    # --- Confidence gate (DO NOT CALL LLM) ---
    if len(subset) < 5:
        return {
            "theme_id": int(theme_id),
            "confidence": "low",
            "insight": None,
            "evidence": evidence_texts,
            "disclaimer": "AI-assisted insight. Final decisions require human review.",
            "message": "Insufficient evidence to generate a reliable AI insight."
        }

    # --- Cache ---
    if "insight_cache" not in STATE:
        STATE["insight_cache"] = {}
    cache_key = f"{int(theme_id)}:{len(subset)}:{n}"
    if (not force) and cache_key in STATE["insight_cache"]:
        return STATE["insight_cache"][cache_key]

    # --- Optional: narrow to dominant product_area to reduce tokens ---
    top_area = subset["product_area"].value_counts().idxmax()
    subset_for_llm = subset[subset["product_area"] == top_area].copy()
    evidence_texts = subset_for_llm.head(min(n, len(subset_for_llm)))["text"].tolist()

    # --- LLM call with safe fallback (NO 503) ---
    try:
        insight = generate_insight_from_evidence_openrouter(evidence_texts)
        payload = {
            "theme_id": int(theme_id),
            "confidence": "high" if len(subset) >= 20 else "medium",
            "insight": insight,
            "evidence": evidence_texts,
            "disclaimer": "AI-assisted insight. Final decisions require human review."
        }
        STATE["insight_cache"][cache_key] = payload
        return payload

    except Exception as e:
        print("INSIGHT ERROR TRACEBACK:")
        print(traceback.format_exc())
        # Return 200 with a graceful fallback so UI still works
        return {
            "theme_id": int(theme_id),
            "confidence": "medium" if len(subset) >= 10 else "low",
            "insight": None,
            "evidence": evidence_texts,
            "disclaimer": "AI-assisted insight. Final decisions require human review.",
            "error_type": "llm_failed",
            "message": "AI insight unavailable (temporary error)."
        }

        
@app.post("/load_demo")
def load_demo(request: Request):
    global DEMO_STATE

    rate_limit(request.client.host, limit=5, window=60)

    if DEMO_STATE is None:
        raise HTTPException(
            status_code=500,
            detail="Demo dataset not available on server."
        )

    STATE.clear()
    STATE.update(copy.deepcopy(DEMO_STATE))

    return {
        "message": "Demo loaded",
        "rows_loaded": int(len(STATE["df"])) if STATE["df"] is not None else 0,
        "num_themes": int(STATE["df"]["theme_id"].nunique()) if STATE["df"] is not None else 0,
    }