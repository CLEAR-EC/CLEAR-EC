
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

## Setup and Installation

### Requirements
- Python 3.9+
- CUDA 12.1 compatible GPU (for faster inference)
- Dependencies listed in `requirements.txt` or `environment.yml`

### Installation

**Option 1: Using conda**
```bash
conda env create -f code/environment.yml
conda activate CLEAR-EC
```

**Option 2: Using pip**
```bash
pip install -r code/requirements.txt
```

---

## How to Run

### 1. Segmentation Pipeline (`main.py`)

The main script performs cell segmentation on corneal endothelial images and computes the three key metrics: Cell Density (CD), Coefficient of Variation (CV), and Hexagonality (HEX).

**Basic Usage:**
```bash
cd code
python main.py --split test
```

**Common Arguments:**
- `--split` (str): Which dataset split to process. Options: `train`, `test` (default: `test`)
- `--data_dir` (str): Directory containing input images. If not specified, defaults to `<repo>/data/<split>_mha`
- `--results_dir` (str): Output directory for predictions CSV (default: `./results_mha`)
- `--vis_output_dir` (str): Directory for segmentation visualization PNGs (default: `./results_mha/visualizations`)
- `--plot`: Save segmentation overlay visualizations (optional flag)
- `--limit` (int): Process only the first N images for quick testing (default: 0, meaning all images)
- `--seed` (int): Random seed for reproducibility (default: 42)

**Cellpose Model Parameters:**
- `--model_type` (str): Cellpose model. Options: `cyto` (default), `cyto2`, `nuclei`
- `--diameter` (float): Average cell diameter in pixels. If not specified, auto-estimates per image (recommended to set explicitly for endothelial cells)
- `--flow_threshold` (float): Cellpose flow threshold (default: 0.4)
- `--cellprob_threshold` (float): Cell probability threshold (default: 0.0)
- `--min_size` (int): Minimum cell size in pixels (default: 15)
- `--no_tile`: Disable internal tiling and run whole-image inference (may cause OOM on large images)

**Random Crop Parameters:**
- `--random_crop_frac` (float): Fraction of image height/width to keep for metric calculation (default: 0.4)

**Example: Process test split with visualizations**
```bash
python main.py --split test --plot --seed 42
```

**Example: Process with custom model and diameter**
```bash
python main.py --split test --model_type cyto2 --diameter 30 --plot
```

**Example: Quick smoke test on first 5 images**
```bash
python main.py --split test --limit 5 --plot
```

**Output:**
- Predictions CSV: `./results_mha/predictions_test.csv` (contains ID, CD, CV, HEX columns)
- Visualizations (if `--plot` enabled): PNG files in `./results_mha/visualizations/`

---

### 2. Evaluation Pipeline (`evaluate.py`)

Compares your predictions against ground-truth metrics and reports per-image errors and average performance.

**Basic Usage:**
```bash
cd code
python evaluate.py --split test
```

**Arguments:**
- `--split` (str): Dataset split to evaluate. Options: `train`, `test` (default: `test`)
- `--predictions_csv` (str): Path to predictions CSV from `main.py` (default: `./results_mha/predictions_test.csv`)
- `--gt_csv` (str): Path to ground-truth CSV with reference metrics
- `--results_dir` (str): Output directory for error analysis CSVs (default: `./results_mha`). Pass empty string to skip writing.

**Example: Evaluate test predictions against ground truth**
```bash
python evaluate.py --split test --gt_csv /path/to/ground_truth.csv
```

**Output:**
- Per-slide error metrics: `./results_mha/errors_test.csv`
- Average error summary: `./results_mha/avg_error_test.csv`
- Prints to console: per-case metrics, per-slide ranking, and average percent error

---

## Complete Workflow Example

```bash
cd code

# 1. Run segmentation on test split
python main.py --split test --plot --seed 42

# 2. Evaluate against ground truth
python evaluate.py --split test --gt_csv /path/to/final_test_ids.csv
```

This will:
1. Segment all images in the test split
2. Save predictions to `./results_mha/predictions_test.csv`
3. Generate visualization PNGs in `./results_mha/visualizations/`
4. Compare predictions against ground truth and report errors

---

## Citation

If you use this repository or adapt this baseline for challenge participation, please cite the CLEAR-EC challenge materials when available.

```bibtex
@misc{clear_ec_baseline,
  title={CLEAR-EC Baseline: Corneal Learning for Endothelial Assessment and Review using AI for Endothelial Count},
  author={CLEAR-EC Challenge Organizers},
  year={2026}
}
