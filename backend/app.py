from fastapi import FastAPI, UploadFile, File, HTTPException, Query
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
import time
import json
from google import genai
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Feedback Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory store (MVP). Resets when server restarts.
STATE = {
    "df": None,
    "vectorizer": None,
    "nn_model": None,
    "tfidf_matrix": None,
    "kmeans": None,
    "theme_labels": None,
    "insight_cache": {}
}

if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not set")
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))



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


def generate_insight_from_evidence_gemini(evidence_texts):
    prompt = f"""You are assisting a product manager.
You MUST use only the evidence provided.
Do NOT invent features, dates, campaigns, or UI elements not explicitly mentioned in evidence.
If evidence spans multiple unrelated requests, choose the single most common/central problem and ignore the rest.

EVIDENCE (verbatim feedback items):
{evidence_texts}

Return ONLY valid JSON with exactly these keys:
title: string (8-12 words, no dates/holidays)
summary: string (2-3 sentences, must cite what users said)
recommended_action: string (1-3 actions, each must be directly supported by evidence)
success_metric: string (must be a measurable metric; if unclear, propose a reasonable proxy metric tied to the evidence, e.g., time-to-complete-task, support tickets, feature adoption)
risks: string (must list 2-3 realistic risks: misclassification, overfitting to internal feedback, UX regression, measurement noise)
evidence_refs: array of integers (indices of evidence items used, 0-based)
"""

    last_err = None
    for attempt in range(2):  # 2 tries max
        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                # If your SDK supports config, add:
                # config={"temperature": 0.2, "max_output_tokens": 500}
            )
            text = resp.text.strip()

            if "```" in text:
                text = text.split("```")[1].strip()
                if text.lower().startswith("json"):
                    text = text[4:].strip()

            return json.loads(text)
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise last_err


@app.get("/health")
def health():
    return {"status": "ok", "rows_loaded": 0 if STATE["df"] is None else int(len(STATE["df"]))}


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

@app.get("/themes/{theme_id}/insight")
def theme_insight(theme_id: int, n: int = 6):
    if STATE["df"] is None:
        raise HTTPException(status_code=400, detail="No data available. Upload first.")

    # Cache hit
    cached = STATE.get("insight_cache", {}).get(int(theme_id))
    if cached:
        return {
            "theme_id": int(theme_id),
            "confidence": "high",
            "insight": cached["insight"],
            "evidence": cached["evidence"],
            "disclaimer": "AI-assisted insight. Final decisions require human review."
        }

    df = STATE["df"]
    subset = df[df["theme_id"] == theme_id].copy()
    if subset.empty:
        raise HTTPException(status_code=404, detail="Theme not found")

    if len(subset) < 5:
        evidence = subset.head(min(n, len(subset)))["text"].tolist()
        return {
            "theme_id": int(theme_id),
            "confidence": "low",
            "insight": None,
            "message": "insufficient evidence",
            "evidence": evidence,
            "disclaimer": "AI-assisted insight. Final decisions require human review."
        }

    top_area = subset["product_area"].value_counts().idxmax()
    subset_for_llm = subset[subset["product_area"] == top_area].copy()

    evidence_texts = subset_for_llm.head(n)["text"].tolist()

    try:
        insight = generate_insight_from_evidence_gemini(evidence_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())

    # Guardrails
    if str(insight.get("success_metric", "")).strip().lower() in ("", "insufficient evidence"):
        insight["success_metric"] = "Reduce related support tickets and increase weekly usage of the affected feature."
    if str(insight.get("risks", "")).strip().lower() in ("", "insufficient evidence"):
        insight["risks"] = "Over-weighting a subset of feedback; UX regression; measurement noise."

    STATE.setdefault("insight_cache", {})[int(theme_id)] = {
        "ts": time.time(),
        "insight": insight,
        "evidence": evidence_texts
    }

    return {
        "theme_id": int(theme_id),
        "confidence": "high",
        "insight": insight,
        "evidence": evidence_texts,
        "disclaimer": "AI-assisted insight. Final decisions require human review."
    }


@app.post("/load_demo")
def load_demo(k: int = 10):
    try:
        demo_path = Path(__file__).parent / "data" / "demo.csv"
        if not demo_path.exists():
            raise RuntimeError(f"demo.csv not found at {demo_path}")

        raw_text = demo_path.read_text(encoding="utf-8", errors="replace")

        df = pd.read_csv(
            StringIO(raw_text),
            sep=None,            # auto-detect delimiter
            engine="python",
            quotechar='"',
            escapechar="\\",
        )

        df_clean = clean_df(df)

        STATE["df"] = df_clean
        texts = df_clean["text"].tolist()

        vectorizer, tfidf_matrix, nn_model = build_embeddings_and_index(texts)
        STATE["vectorizer"] = vectorizer
        STATE["tfidf_matrix"] = tfidf_matrix
        STATE["nn_model"] = nn_model

        km, theme_ids = run_kmeans(tfidf_matrix, k)
        STATE["kmeans"] = km

        STATE["df"] = STATE["df"].copy()
        STATE["df"]["theme_id"] = theme_ids.astype(int)
        STATE["df"]["sentiment_label"] = "NEUTRAL"
        STATE["df"]["sentiment_score"] = 0.0

        STATE["theme_labels"] = label_themes_tfidf(
            STATE["df"]["text"].tolist(),
            theme_ids,
            topn=5
        )

        return {"message": "Demo loaded", "rows_loaded": int(len(df_clean)), "num_themes": int(k)}

    except Exception:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=tb)
