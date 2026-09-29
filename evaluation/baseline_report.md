# Clustering Evaluation Baseline

## Scope

The clustering model is fitted on all 150 feedback rows. External metrics use the 20 rows with human-reviewed primary themes. These results are directional and intended for POC iteration, not a production accuracy claim.

## Result

The strongest tested configuration is **k=9** with a directional score of **0.412**.
The current application uses **k=10**, with a directional score of **0.399**.

The directional score combines adjusted Rand index, normalized mutual information, pairwise F1, and silhouette score. Purity and theme cohesion are diagnostic only because they can reward excessive cluster counts.

## Metrics by cluster count

| k | ARI | NMI | Pair precision | Pair recall | Pair F1 | Purity | Theme cohesion | Silhouette | Directional |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.020 | 0.497 | 0.064 | 0.300 | 0.105 | 0.300 | 0.750 | 0.011 | 0.282 |
| 5 | 0.016 | 0.522 | 0.061 | 0.300 | 0.102 | 0.400 | 0.750 | 0.011 | 0.286 |
| 6 | 0.086 | 0.524 | 0.098 | 0.600 | 0.169 | 0.300 | 0.850 | 0.009 | 0.321 |
| 7 | 0.052 | 0.597 | 0.083 | 0.300 | 0.130 | 0.450 | 0.750 | 0.013 | 0.322 |
| 8 | 0.131 | 0.644 | 0.128 | 0.500 | 0.204 | 0.450 | 0.800 | 0.016 | 0.372 |
| **9** | 0.167 | 0.754 | 0.176 | 0.300 | 0.222 | 0.600 | 0.750 | 0.013 | 0.412 |
| 10 | 0.148 | 0.727 | 0.158 | 0.300 | 0.207 | 0.550 | 0.750 | 0.026 | 0.399 |
| 11 | 0.087 | 0.697 | 0.107 | 0.300 | 0.158 | 0.600 | 0.750 | 0.025 | 0.364 |
| 12 | 0.157 | 0.734 | 0.167 | 0.300 | 0.214 | 0.550 | 0.700 | 0.015 | 0.403 |
| 13 | 0.157 | 0.747 | 0.167 | 0.300 | 0.214 | 0.600 | 0.750 | 0.016 | 0.406 |
| 14 | 0.142 | 0.686 | 0.135 | 0.500 | 0.213 | 0.550 | 0.800 | 0.019 | 0.387 |
| 15 | 0.126 | 0.695 | 0.129 | 0.400 | 0.195 | 0.500 | 0.800 | 0.012 | 0.381 |

## Reviewed rows at k=10

| Feedback ID | Human theme | Predicted cluster |
|---:|---|---:|
| 1 | `editor_reliability` | 9 |
| 2 | `automation_workflows` | 9 |
| 3 | `marketing_initiative` | 5 |
| 4 | `onboarding` | 1 |
| 5 | `mobile_ui` | 7 |
| 6 | `analytics` | 0 |
| 8 | `collaboration` | 6 |
| 9 | `editor` | 0 |
| 10 | `dashboard` | 0 |
| 11 | `analytics` | 5 |
| 12 | `automation_workflows` | 9 |
| 13 | `dashboard` | 4 |
| 17 | `versioning` | 7 |
| 22 | `permissions` | 1 |
| 34 | `search` | 3 |
| 35 | `analytics` | 5 |
| 44 | `permissions` | 4 |
| 46 | `dashboard` | 4 |
| 51 | `search` | 1 |
| 70 | `mobile_ui` | 9 |

## Interpretation limits

- The reviewed sample is small and contains several themes represented by only one example.
- Scores can identify obvious regressions and guide the next experiment, but small differences are not conclusive.
- Cluster IDs are arbitrary. The metrics compare grouping structure rather than cluster numbers.
- Human review remains necessary for final theme names and product decisions.
