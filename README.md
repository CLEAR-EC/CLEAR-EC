
# CLEAR-EC Challenge

**CLEAR-EC**: **C**orneal **L**earning for **E**ndothelial **A**ssessment and **R**eview using AI for **E**ndothelial **C**ount

This repository provides a **baseline pipeline** for the CLEAR-EC challenge. The goal of this baseline is to estimate clinically relevant endothelial cell assessment metrics from corneal endothelial images:

1. **Cell segmentation**
2. **Instance-level cell analysis**
3. **Metric calculation**
   - **Cell density**
   - **Coefficient of variation (CV)**
   - **Hexagonality**

This baseline is intended as a transparent starting point for challenge participants. It is not expected to be the strongest-performing solution, but rather a reference implementation showing how raw images can be converted into challenge outputs.

---

## Overview

Corneal endothelial assessment is important for evaluating corneal graft quality. In this baseline, we formulate the task as:

- **Input**: Corneal image
- **Intermediate output**: Segmented endothelial cells
- **Final output**: Quantitative metrics

The baseline pipeline first segments cells, then extracts cell morphology from the predicted masks, and finally computes the key metrics used in corneal image assessment.

---

## Baseline Pipeline

### Step 1: Cell Segmentation
The first stage performs **cell segmentation** on the corneal endothelial image.  
The purpose of this stage is to identify individual endothelial cells and obtain cell masks.

### Step 2: Cell Morphology Extraction
From the segmentation result, the pipeline derives per-cell geometric properties such as:

- Cell area
- Cell perimeter
- Number of polygon sides / neighbors
- Cell centroid

These features are then used to compute clinically meaningful summary metrics.

### Step 3: Metric Calculation
The baseline computes the following metrics:

#### 1. Cell Density
Cell density measures the number of endothelial cells per unit area.

#### 2. Coefficient of Variation (CV)
Coefficient of variation reflects variation in cell area.

#### 3. Hexagonality
Hexagonality measures the proportion of cells that are hexagonal. A cell is considered hexagonal if it has six sides or six neighbors under the polygon approximation or adjacency graph used in the implementation.

---

## Repository Purpose

This repository is designed to:

- Provide a **reference baseline** for the CLEAR-EC challenge
- Show a simple end-to-end path from image to evaluation metrics
- Offer a starting point for participants to build stronger methods

---

## Expected Input and Output

### Input
The baseline assumes input images are corneal endothelial microscopy images provided by the challenge.

### Output
For each image, the baseline is expected to output:

- Predicted cell segmentation
- Derived endothelial assessment metrics:
  - Cell density
  - Coefficient of variation
  - Hexagonality

Depending on the challenge submission format, outputs may be saved as:

- Segmentation masks
- CSV / JSON files containing the computed metrics

---

## Notes on the Baseline

This baseline is intentionally simple. Participants are encouraged to improve upon this baseline by developing better segmentation models, more robust post-processing, and more accurate morphology estimation methods.

---

## Challenge Context

CLEAR-EC focuses on AI methods for **corneal endothelial assessment and review**, with an emphasis on clinically meaningful quantitative endpoints derived from endothelial cell structure.

This baseline reflects a classical analysis pipeline:

**Image → Cell Segmentation → Cell Morphology → Clinical Metrics**

which offers interpretability and direct linkage between model predictions and downstream endothelial measurements.

---

## Citation

If you use this repository or adapt this baseline for challenge participation, please cite the CLEAR-EC challenge materials when available.

```bibtex
@misc{clear_ec_baseline,
  title={CLEAR-EC Baseline: Corneal Learning for Endothelial Assessment and Review using AI for Endothelial Count},
  author={CLEAR-EC Challenge Organizers},
  year={2026}
}
