# Diabetic Retinopathy Severity Grading with Uncertainty Quantification

BTech coursework / portfolio project on diabetic retinopathy (DR) severity
classification from retinal fundus images, with a focus on **uncertainty
quantification** as the key differentiator over a plain classifier.

## Problem

Diabetic retinopathy is graded by clinicians into five severity classes:

0. No DR
1. Mild
2. Moderate
3. Severe
4. Proliferative DR

Standard CNN classifiers report a class label but not how confident they
are — a dangerous gap in a clinical setting, where a wrong prediction made
with high confidence is worse than one flagged as uncertain and deferred to
a human. This project builds a DR grading model that also knows what it
doesn't know, and evaluates that uncertainty rigorously rather than just
reporting top-line accuracy.

## Dataset

[APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
(Kaggle) — retinal fundus photographs labeled with the 5-class DR severity
scale above.

## Plan

1. **Baseline classifier** — fine-tune a pretrained CNN (EfficientNet or
   ResNet) on APTOS 2019 for 5-class DR severity grading.
2. **Uncertainty estimation** — add Monte Carlo Dropout (MC Dropout) to
   obtain predictive uncertainty from the trained model at inference time.
3. **Calibration** — quantify how well predicted confidence matches
   actual correctness: Expected Calibration Error (ECE) and reliability
   diagrams.
4. **Selective prediction** — build an accuracy-vs-coverage curve showing
   accuracy when the model is allowed to defer (abstain on) the most
   uncertain X% of cases.
5. **Interpretability** — Grad-CAM visualizations to inspect which image
   regions drive each prediction.

## Project structure

```
data/
  raw/          # original APTOS images + labels (not committed)
  processed/    # preprocessed/split data ready for training

src/
  data_loader.py   # dataset loading, preprocessing, train/val/test splits
  model.py         # model architecture, MC Dropout wiring
  calibration.py   # ECE, reliability diagrams, calibration utilities
  eval.py          # metrics, selective prediction / accuracy-coverage curve
  gradcam.py       # Grad-CAM interpretability

notebooks/     # exploratory analysis only — no production logic

results/
  metrics/     # saved metric outputs (accuracy, ECE, etc.)
  plots/       # reliability diagrams, Grad-CAM overlays, selective
               # prediction curves
```

## Status

Project scaffolding only — no model code yet.
