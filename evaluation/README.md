# Feedback clustering evaluation

This folder contains a small human-reviewed benchmark for evaluating the dashboard's unsupervised theme clustering.

## Run the baseline

From the repository root:

```bash
python evaluation/evaluate_clustering.py
```

The script fits the current TF-IDF and MiniBatchKMeans approach on all feedback rows, evaluates cluster counts from 4 to 15 against the reviewed subset, and writes `baseline_report.md`.

## How to interpret it

The reviewed sample is intentionally small because this is an MVP evaluation. Use the scores to compare controlled experiments and catch clear regressions. Do not present small score differences as production-grade accuracy.

The report includes:

- adjusted Rand index and normalized mutual information, which compare grouping structure without requiring cluster IDs to match theme names;
- pairwise precision, recall, and F1, which check whether feedback with the same human theme stays together;
- silhouette score, which measures separation across the full dataset;
- purity and theme cohesion as diagnostic measures.

The combined directional score excludes purity and cohesion because both can be inflated by creating more clusters.
