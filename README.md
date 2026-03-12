# RNA-seq Caffeine Tolerance ML Project

![theme-yellow](https://img.shields.io/badge/theme-yellow-F7C948)
![theme-blue](https://img.shields.io/badge/theme-blue-2D6CDF)
![theme-pink](https://img.shields.io/badge/theme-pink-FF5CA8)

<p>
  <strong>Color palette:</strong> yellow + blue + pink<br/>
  <strong>Focus:</strong> ML practice with RNA-seq + caffeine biology + nutrition-inspired extension
</p>

---

## Why This Project

This project puts machine learning into practice on RNA sequencing data while clearly contrasting:
- **Supervised learning**: predict known labels
- **Unsupervised learning**: discover latent structure without labels

The biological storyline is caffeine response, with an extension toward nutrition context (including sugar-related reward/dopamine questions).

## Main Question
Can RNA expression profiles capture meaningful caffeine-associated signals, and what is learned differently by supervised vs unsupervised methods?

## Selected Dataset
- Primary dataset: `GSE167121` (mouse hippocampus RNA-seq, caffeine vs water)
- Dataset notes and metadata constraints: `reports/dataset_selection.md`

---

## Interactive Dashboard (New)

The project includes an interactive simulator at `dashboard/app.py` where users can tune:
- number of cups
- caffeine per cup
- cup timing and spacing
- half-life preset (fast/average/slow/custom)
- effect profile (focused, balanced, sensitive)
- sugar with coffee toggle

Dashboard outputs:
- active caffeine decay curve
- effect indices over time (`focus`, `jitter`, `mood`)
- summary cards (peak caffeine, evening residual, etc.)

Run it:
```bash
streamlit run dashboard/app.py
```

---

## Repository Layout
- `data/raw/`: raw downloaded expression and metadata files
- `data/processed/`: cleaned matrices used for modeling
- `notebooks/`: guided analysis notebooks
- `src/`: reusable preprocessing, modeling, evaluation, and plotting code
- `dashboard/`: interactive caffeine simulator
- `reports/`: dataset notes and interpretation summaries

## ML Modules
- **Unsupervised**
  - PCA and UMAP embeddings
  - k-means and hierarchical clustering
  - silhouette score and post hoc cluster-label purity
- **Supervised**
  - logistic regression and random forest baselines
  - subject-aware CV when `subject_id` is available
  - metrics: accuracy, F1, ROC-AUC, confusion matrix
  - top feature interpretation

---

## Quick Start
1. Create environment and install dependencies:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Place expression and metadata files in `data/raw/`.
3. Run notebooks in order:
   - `notebooks/01_eda.ipynb`
   - `notebooks/02_unsupervised.ipynb`
   - `notebooks/03_supervised.ipynb`
   - `notebooks/04_comparison.ipynb`
4. Optional: launch dashboard:
   - `streamlit run dashboard/app.py`

## Input Schema (Expected)
### Expression matrix
- Rows: samples
- Columns: genes
- Required key: `sample_id`

### Metadata table
- Required:
  - `sample_id`
  - `condition` (`caffeine` / `water` or equivalent)
- Optional:
  - `subject_id` (group-aware split logic)
  - `timepoint` (pre/post)
  - `nutrition_group` (fed/fasted/diet)
  - `response_label` (tolerance proxy)

## Nutrition + Dopamine Extension
MVP uses one caffeine RNA-seq dataset for reproducibility. Next step is integrating a nutrition-focused dataset and comparing pathway-level signatures linked to reward/synaptic (dopamine-relevant) biology.
