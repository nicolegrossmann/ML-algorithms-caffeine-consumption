# Dataset Selection

## Primary Dataset (Selected)
- **GEO accession:** `GSE167121`
- **Title:** Genomewide effects of regular caffeine intake on hippocampal metabolism and learning-dependent transcription [RNA-Seq]
- **Species / tissue:** Mouse (`Mus musculus`), hippocampus
- **Intervention:** Chronic caffeine in drinking water (`0.3 g/L`) vs water control
- **Sample count:** 16
- **Processed counts file:** `GSE167121_S17065_RawReadCounts.txt.gz`

This dataset is selected for the MVP because it provides a clean RNA-seq treatment/control setup with enough samples for both unsupervised structure discovery and supervised baseline modeling.

## Metadata Availability (for plan requirements)
- **Condition label (required):** available (`caffeine` vs `water`)
- **Sample identifier (required):** available (GSM sample IDs)
- **Subject identifier (paired design):** not explicitly available as a repeated pre/post subject field
- **Nutrition covariates:** not explicit in this dataset

## Practical adaptation (aligned with plan fallback)
Because true paired pre/post subject IDs are not exposed in this selected dataset, the project uses the fallback supervised target strategy:
1. primary supervised target = `condition` (`caffeine` vs `water`)
2. optional extended target = response context inferred from experiment groups when available in metadata

The codebase keeps a `subject_id` field in the schema so a true paired human/mouse caffeine challenge dataset can be dropped in later without refactoring.

## Nutrition + Dopamine Tie-in (your extension idea)
To connect with your idea ("higher dopamine release with sugar in presence of caffeine"), this project includes an extension pathway:
- keep `GSE167121` as the core caffeine RNA-seq learning dataset
- later add a second nutrition-focused mouse dataset (high-fat/high-sugar context) and perform cross-study comparison
- map convergent pathways/genes related to reward, synaptic plasticity, and dopaminergic signaling in interpretation

This keeps the MVP tractable while preserving your biological story.
