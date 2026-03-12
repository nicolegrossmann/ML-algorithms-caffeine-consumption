# Data Dictionary

## Expression Table
- **File example:** `data/raw/expression_matrix.csv`
- **Grain:** one row per sample
- **Columns:**
  - `sample_id` (string, required)
  - gene columns (numeric counts or normalized values)

## Metadata Table
- **File example:** `data/raw/metadata.csv`
- **Grain:** one row per sample
- **Columns:**
  - `sample_id` (string, required, join key)
  - `condition` (string, required; e.g., caffeine/water)
  - `subject_id` (string, optional; required for paired/group split behavior)
  - `timepoint` (string, optional; e.g., pre/post)
  - `nutrition_group` (string, optional; e.g., fasted/fed/high_sugar)
  - `response_label` (string/int, optional; tolerance or response proxy)
