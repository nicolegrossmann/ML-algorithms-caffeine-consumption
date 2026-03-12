# Supervised vs Unsupervised Comparison

## What unsupervised analysis identifies
Unsupervised methods (PCA, UMAP, clustering) test whether samples naturally organize into separable expression states without using condition labels during fitting.

In this project, they answer:
- do caffeine-exposed samples trend toward distinct latent regions?
- is there cluster structure that could reflect response heterogeneity?
- does optional nutrition metadata align with discovered structure?

Primary unsupervised signals:
- `kmeans_silhouette` and `hier_silhouette` for cluster compactness/separation
- post hoc label purity (`kmeans_label_purity`, `hier_label_purity`) to estimate biological alignment

## What supervised analysis identifies
Supervised models directly optimize prediction of known labels (e.g., `condition`, or a future tolerance proxy). They answer:
- how predictable the caffeine-related label is from expression
- which genes contribute most to classification boundaries

Primary supervised signals:
- logistic regression and random forest metrics (`accuracy`, `f1`, `roc_auc`)
- confusion matrices for error structure
- top weighted/important genes for interpretation

## How to interpret differences
- Strong supervised performance with weak unsupervised separation can mean predictive but subtle distributed signals.
- Strong unsupervised separation with modest supervised metrics can indicate non-linear or subgroup structure not fully captured by baseline classifiers.
- Agreement across both approaches (good clustering and good prediction) supports a robust condition-linked transcriptomic signal.

## Caveats
- Small RNA-seq cohorts can inflate variance in both clustering and classifier metrics.
- In the selected dataset, explicit paired pre/post subject IDs are not available; this limits within-subject causal framing.
- High-dimensional gene space increases overfitting risk; filtering, standardization, and conservative interpretation are essential.
- Batch and technical effects can mimic biology if not checked.

## Nutrition + caffeine + dopamine extension (next phase)
To connect with your interest in sugar+caffeine reward biology:
1. add a nutrition-focused mouse RNA-seq dataset with high-sugar context
2. harmonize shared genes and metadata fields
3. compare pathway-level signatures (reward/synaptic/dopaminergic processes) across caffeine-only vs nutrition+caffeine contexts

This enables a biologically compelling narrative: baseline caffeine transcriptomic effects, then nutrition-modulated reward-related effects.
