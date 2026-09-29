#!/usr/bin/env python3
"""Evaluate clustering against the manually reviewed feedback sample."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

DEFAULT_INPUT = Path(__file__).with_name("feedback_labels.csv")
DEFAULT_REPORT = Path(__file__).with_name("baseline_report.md")


def pairwise_scores(true_labels: list[str], cluster_ids: list[int]) -> dict[str, float]:
    tp = fp = fn = 0
    for left, right in combinations(range(len(true_labels)), 2):
        same_theme = true_labels[left] == true_labels[right]
        same_cluster = cluster_ids[left] == cluster_ids[right]
        if same_theme and same_cluster:
            tp += 1
        elif not same_theme and same_cluster:
            fp += 1
        elif same_theme and not same_cluster:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pairwise_precision": precision, "pairwise_recall": recall, "pairwise_f1": f1}


def majority_share(groups: dict[object, list[object]]) -> float:
    correct = total = 0
    for values in groups.values():
        counts: dict[object, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        correct += max(counts.values())
        total += len(values)
    return correct / total if total else 0.0


def purity_scores(true_labels: list[str], cluster_ids: list[int]) -> tuple[float, float]:
    by_cluster: dict[int, list[str]] = {}
    by_theme: dict[str, list[int]] = {}
    for theme, cluster in zip(true_labels, cluster_ids):
        by_cluster.setdefault(cluster, []).append(theme)
        by_theme.setdefault(theme, []).append(cluster)
    return majority_share(by_cluster), majority_share(by_theme)


def build_matrix(texts: pd.Series):
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2)).fit_transform(texts.astype(str))


def evaluate(df: pd.DataFrame, k_values: range) -> list[dict[str, float]]:
    matrix = build_matrix(df["text"])
    reviewed_mask = df["primary_theme"].fillna("").str.strip().ne("")
    true_labels = df.loc[reviewed_mask, "primary_theme"].tolist()
    results: list[dict[str, float]] = []
    for k in k_values:
        model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=256)
        all_clusters = model.fit_predict(matrix)
        reviewed_clusters = all_clusters[reviewed_mask.to_numpy()].tolist()
        purity, theme_cohesion = purity_scores(true_labels, reviewed_clusters)
        pairs = pairwise_scores(true_labels, reviewed_clusters)
        silhouette = silhouette_score(matrix, all_clusters, metric="cosine")
        ari = adjusted_rand_score(true_labels, reviewed_clusters)
        nmi = normalized_mutual_info_score(true_labels, reviewed_clusters)
        directional_score = np.mean(
            [max(0.0, ari), nmi, pairs["pairwise_f1"], (silhouette + 1.0) / 2.0]
        )
        results.append({
            "k": k, "ari": ari, "nmi": nmi, **pairs, "purity": purity,
            "theme_cohesion": theme_cohesion, "silhouette": silhouette,
            "directional_score": float(directional_score),
        })
    return results


def cluster_review_table(df: pd.DataFrame, k: int) -> list[dict[str, str]]:
    matrix = build_matrix(df["text"])
    clusters = MiniBatchKMeans(
        n_clusters=k, random_state=42, n_init=10, batch_size=256
    ).fit_predict(matrix)
    reviewed_mask = df["primary_theme"].fillna("").str.strip().ne("")
    reviewed = df.loc[reviewed_mask].copy()
    reviewed["cluster_id"] = clusters[reviewed_mask.to_numpy()]
    return [
        {"id": str(row["id"]), "human_theme": row["primary_theme"], "cluster_id": str(int(row["cluster_id"]))}
        for _, row in reviewed.iterrows()
    ]


def render_report(results, review_rows, reviewed_count: int, total_count: int, current_k: int) -> str:
    best = max(results, key=lambda row: row["directional_score"])
    current = next((row for row in results if row["k"] == current_k), None)
    lines = [
        "# Clustering Evaluation Baseline", "", "## Scope", "",
        f"The clustering model is fitted on all {total_count} feedback rows. External metrics use the {reviewed_count} rows with human-reviewed primary themes. These results are directional and intended for POC iteration, not a production accuracy claim.",
        "", "## Result", "",
        f"The strongest tested configuration is **k={int(best['k'])}** with a directional score of **{best['directional_score']:.3f}**.",
    ]
    if current:
        lines += [f"The current application uses **k={current_k}**, with a directional score of **{current['directional_score']:.3f}**.", ""]
    lines += [
        "The directional score combines adjusted Rand index, normalized mutual information, pairwise F1, and silhouette score. Purity and theme cohesion are diagnostic only because they can reward excessive cluster counts.",
        "", "## Metrics by cluster count", "",
        "| k | ARI | NMI | Pair precision | Pair recall | Pair F1 | Purity | Theme cohesion | Silhouette | Directional |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        k_cell = f"**{int(row['k'])}**" if row["k"] == best["k"] else str(int(row["k"]))
        lines.append(
            f"| {k_cell} | {row['ari']:.3f} | {row['nmi']:.3f} | {row['pairwise_precision']:.3f} | "
            f"{row['pairwise_recall']:.3f} | {row['pairwise_f1']:.3f} | {row['purity']:.3f} | "
            f"{row['theme_cohesion']:.3f} | {row['silhouette']:.3f} | {row['directional_score']:.3f} |"
        )
    lines += ["", f"## Reviewed rows at k={current_k}", "", "| Feedback ID | Human theme | Predicted cluster |", "|---:|---|---:|"]
    for row in review_rows:
        lines.append(f"| {row['id']} | `{row['human_theme']}` | {row['cluster_id']} |")
    lines += [
        "", "## Interpretation limits", "",
        "- The reviewed sample is small and contains several themes represented by only one example.",
        "- Scores can identify obvious regressions and guide the next experiment, but small differences are not conclusive.",
        "- Cluster IDs are arbitrary. The metrics compare grouping structure rather than cluster numbers.",
        "- Human review remains necessary for final theme names and product decisions.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-k", type=int, default=4)
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--current-k", type=int, default=10)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    missing = {"id", "text", "primary_theme"} - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    reviewed_count = int(df["primary_theme"].fillna("").str.strip().ne("").sum())
    if reviewed_count < 2:
        raise SystemExit("At least two reviewed rows are required")
    results = evaluate(df, range(args.min_k, args.max_k + 1))
    review_rows = cluster_review_table(df, args.current_k)
    args.report.write_text(render_report(results, review_rows, reviewed_count, len(df), args.current_k), encoding="utf-8")
    best = max(results, key=lambda row: row["directional_score"])
    print(f"Reviewed rows: {reviewed_count}/{len(df)}")
    print(f"Best tested k: {int(best['k'])}")
    print(f"Directional score: {best['directional_score']:.3f}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
