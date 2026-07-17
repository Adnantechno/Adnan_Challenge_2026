# Web Attack Detection on CIC-IDS-2017

**CSLab Challenge 2026 — Adnan**

Detecting web attacks (Brute Force, XSS, SQL Injection) in the CIC-IDS-2017
dataset using a three-layer machine learning system with explainability.

The web attacks in this dataset were generated against **DVWA** — a deliberately
vulnerable PHP/MySQL web application acting as the victim — with XSS attacks
automated via Selenium. The goal of this challenge is to **detect these attacks
from the recorded network flows**.

---

## Approach

Three complementary detection layers, plus SHAP for explainability:

| Layer | Technique | Type | Detects |
|-------|-----------|------|---------|
| 1 | **XGBoost** | Supervised (per-flow) | Known web-attack patterns |
| 2 | **Isolation Forest** | Unsupervised anomaly (per-flow) | Novel / abnormal flows |
| 3 | **CRF** | Probabilistic Graphical Model (sequence) | Attacks in their temporal sequence |

## Dataset

- CIC-IDS-2017 — `Thursday-WorkingHours-Morning-WebAttacks`
  (GeneratedLabelledFlows version, which retains timestamps for sequence modelling).
- 170,231 clean flows x 77 numeric features after cleaning.
- Severe class imbalance (only 1.28% attacks), handled with SMOTE and evaluated
  with F1 / Recall rather than accuracy.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| XGBoost (supervised) | 0.9997 | 0.9789 | 0.9954 | 0.9871 |
| Isolation Forest (anomaly) | 0.9751 | 0.0469 | 0.0489 | 0.0479 |
| Combined (OR) | 0.9869 | 0.4947 | 0.9954 | 0.6609 |
| CRF (sequence / PGM) | 0.9934 | 0.9098 | 0.7873 | 0.8441 |

XGBoost ROC-AUC = 1.000. Top attack-signalling features (via SHAP): TCP window
sizes (`Init_Win_bytes_backward` / `forward`), `min_seg_size_forward`,
`URG Flag Count`, and packet-timing features (`Flow IAT Min`).

## Files in this folder

| File | Description |
|------|-------------|
| `WebAttack_Presentation_White.pptx` | The 5-slide presentation (approach, results, explainability) |
| `WebAttack_Detection_with_CRF.ipynb` | Full analysis notebook — all code, models and result figures |
| `resulting_dataset_perflow.csv` | Resulting dataset: per-flow predictions (XGBoost, Isolation Forest, Combined) on the test set |
| `resulting_dataset_CRF.csv` | Resulting dataset: CRF sequence-model predictions (time-ordered flows) |
| `results_all_models.csv` | Summary of performance metrics for every model |
| `app.py` | Streamlit detection console (optional live demo) |

## Two resulting datasets — why?

The per-flow models (XGBoost, Isolation Forest, Combined) predict on individual
flows, so their outputs share one table (`resulting_dataset_perflow.csv`).

The CRF is a **sequence** model: it uses discretised features, time-ordered flows,
and a blocked train/test split, so its predictions correspond to a different set
of flows in a different order. Its output is therefore provided separately in
`resulting_dataset_CRF.csv`.

## Python tools & libraries

`pandas` · `scikit-learn` · `XGBoost` · `imbalanced-learn` (SMOTE) ·
`sklearn-crfsuite` · `SHAP` · `Streamlit` · `matplotlib` / `seaborn`

## Methodology notes

- Leak-free pipeline: split -> scale -> SMOTE (training set only).
- Per-flow models use a stratified train/test split; the CRF uses a blocked split
  so attacks appear in both the training and test portions.
- Metadata columns (IPs, ports, flow ID, timestamp) were excluded from the feature
  set so the model learns behaviour rather than attacker identity; the timestamp
  was used only to order flows for the sequence layer.

## How to run the app (optional)

```
pip install streamlit xgboost scikit-learn sklearn-crfsuite shap joblib
streamlit run app.py
```
