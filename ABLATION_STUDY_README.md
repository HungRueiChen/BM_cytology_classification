# Ablation Study: Random Tile Selection - Usage Guide

## Overview

This guide explains how to use the ablation study scripts to compare quality-based tile selection against random tile selection.

## Scripts Created

### 1. Tile Selection Script (Random & Quality-Based)
**File**: [`4_choose_200_random_ROIs_single_wsi.py`](file:///home/user/16tb2/BM_cytology_classification/BM_cytology_classification/4_choose_200_random_ROIs_single_wsi.py)

Directly tiles WSI files using OpenSlide and selects 200 tiles using either random or quality-based selection.

**Key Features**:
- **Dual Mode Support**: Random selection or quality-based selection
- Tiles WSI files (.mrxs) directly using OpenSlide
- Filters out background tiles (all-white or all-black corners) in random mode
- Uses quality scores from CSV files in quality mode
- Saves only the selected 200 tiles (no intermediate storage)
- Memory efficient - processes one WSI at a time

**Selection Strategies**:
- **Random Mode**: Randomly samples from valid (non-background) tiles
- **Quality Mode**: 
  - Prefers tiles with quality score ≥ 0.8
  - Falls back to 0.5-0.8 range if insufficient high-quality tiles
  - Reads quality scores from CSV files

**Usage Examples**:

```bash
# Random selection (default)
python 4_choose_200_random_ROIs_single_wsi.py \
    --mode random \
    --slide_id TV0001 \
    --label ALL \
    --group training \
    --wsi_source_dir ../Bone_marrow_cytology_WSI \
    --dest_parent_dir ../Final_BM_cytology_all_200_random

# Quality-based selection
python 4_choose_200_random_ROIs_single_wsi.py \
    --mode quality \
    --slide_id TV0001 \
    --label ALL \
    --group training \
    --wsi_source_dir ../Bone_marrow_cytology_WSI \
    --quality_csv_dir ../BM_cytology_tile_select/0523_VGG16_20e_0001_then_180e_00001_SGD/all_probs \
    --dest_parent_dir ../Final_BM_cytology_all_200_quality

# Process all WSIs from cohort JSON (random mode)
python 4_choose_200_random_ROIs_single_wsi.py \
    --mode random \
    --process_all \
    --cohort_json ../Final_BM_cytology_all_200/1202_cohort.json \
    --wsi_source_dir ../Bone_marrow_cytology_WSI \
    --dest_parent_dir ../Final_BM_cytology_all_200_random

# Process all WSIs from cohort JSON (quality mode)
python 4_choose_200_random_ROIs_single_wsi.py \
    --mode quality \
    --process_all \
    --cohort_json ../Final_BM_cytology_all_200/1202_cohort.json \
    --wsi_source_dir ../Bone_marrow_cytology_WSI \
    --quality_csv_dir ../BM_cytology_tile_select/0523_VGG16_20e_0001_then_180e_00001_SGD/all_probs \
    --dest_parent_dir ../Final_BM_cytology_all_200_quality
```

### 2. Pipeline Orchestration Script
**File**: [`ablation_study_pipeline.py`](file:///home/user/16tb2/BM_cytology_classification/BM_cytology_classification/ablation_study_pipeline.py)

Coordinates the entire ablation study workflow.

**Usage Examples**:

```bash
# Run complete pipeline (all steps)
python ablation_study_pipeline.py \
    --step all \
    --cohort_json ../Final_BM_cytology_all_200/1202_cohort.json \
    --architecture DenseNet121 \
    --epochs 100

# Run only tile selection
python ablation_study_pipeline.py \
    --step select \
    --cohort_json ../Final_BM_cytology_all_200/1202_cohort.json

# Run only training
python ablation_study_pipeline.py \
    --step train \
    --train_dataset ../Final_BM_cytology_all_200_random \
    --exp_name my_ablation_experiment \
    --architecture DenseNet121 \
    --epochs 100 \
    --batch_size 8

# Run only testing
python ablation_study_pipeline.py \
    --step test \
    --test_dataset ../Final_BM_cytology_all_200_random/test \
    --exp_dir ../BM_cytology_classification_logs/my_ablation_experiment

# Run only comparison
python ablation_study_pipeline.py \
    --step compare \
    --quality_results_dir ../quality_based_results \
    --random_results_dir ../random_based_results \
    --comparison_output_dir ./ablation_comparison
```

### 3. Results Comparison Script
**File**: [`compare_ablation_results.py`](file:///home/user/16tb2/BM_cytology_classification/BM_cytology_classification/compare_ablation_results.py)

Compares results and generates visualizations and statistical analysis.

**Usage Example**:

```bash
python compare_ablation_results.py \
    --quality_dir ../quality_based_results \
    --random_dir ../random_based_results \
    --output_dir ./ablation_comparison
```

**Outputs**:
- `comparison_table.csv` - Side-by-side metric comparison
- `metric_comparison.png` - Bar charts comparing metrics
- `improvement_heatmap.png` - Heatmap showing percentage improvements
- `roc_comparison.png` - ROC curve comparisons
- `statistical_tests.csv` - Statistical significance tests
- `ablation_summary_report.txt` - Comprehensive text summary

## Complete Workflow

### Step 1: Random Tile Selection

```bash
python 4_choose_200_random_ROIs_single_wsi.py \
    --process_all \
    --cohort_json ../Final_BM_cytology_all_200/1202_cohort.json \
    --wsi_source_dir ../Bone_marrow_cytology_WSI \
    --dest_parent_dir ../Final_BM_cytology_all_200_random
```

This directly tiles WSI files and creates the directory structure:
```
../Final_BM_cytology_all_200_random/
├── training/
│   ├── ALL/
│   ├── AML_APL/
│   ├── CML/
│   ├── Lymphoma_CLL/
│   └── MM/
├── validation/
│   └── [same structure]
├── test/
│   └── [same structure]
└── 1202_cohort_tiles_random.json
```

### Step 2: Train Classification Model

```bash
python Ablation_Classification.py \
    ../Final_BM_cytology_all_200_random \
    ablation_random_densenet121 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --freeze_epochs 10 \
    --fine_tune_epochs 90
```

> [!NOTE]
> `Ablation_Classification.py` is a simplified version that uses only DenseNet121 and removes SLURM job submission.

### Step 3: Test the Model

```bash
python New_Testing.py \
    ../Final_BM_cytology_all_200_random/test \
    ../BM_cytology_classification_logs/ablation_random_densenet121 \
    --batch_size 8
```

This generates:
- `test_metrics.json` - Detailed metrics
- `test_metrics.csv` - Summary metrics table
- Confusion matrices and ROC curves

### Step 4: Compare Results

```bash
python compare_ablation_results.py \
    --quality_dir ../BM_cytology_classification_logs/quality_based_experiment \
    --random_dir ../BM_cytology_classification_logs/ablation_random_densenet121 \
    --output_dir ./ablation_comparison
```

## Key Features

### Direct WSI Tiling
- Tiles WSI files (.mrxs) directly using OpenSlide
- No need for pre-tiled images
- Filters background tiles automatically
- Saves only selected tiles (minimal storage)

### Background Filtering
- Excludes tiles where all corner pixels are white (#FFF) or black (#000)
- Ensures only tissue-containing tiles are selected
- Same filtering logic as [`WSI_OpenSlide_V4.py`](file:///home/user/16tb2/BM_cytology_classification/BM_cytology_classification/WSI_OpenSlide_V4.py)

### Compatibility
- Uses [`Ablation_Classification.py`](file:///home/user/16tb2/BM_cytology_classification/BM_cytology_classification/Ablation_Classification.py) for training (simplified, DenseNet121 only)
- Works with existing [`New_Testing.py`](file:///home/user/16tb2/BM_cytology_classification/BM_cytology_classification/New_Testing.py) for evaluation
- Maintains same directory structure as quality-based approach
- Uses same cohort JSON format

### Flexibility
- Can process single WSI or batch mode
- Configurable number of tiles (default 200)
- Supports all disease labels and dataset splits

## Expected Results

The ablation study should demonstrate that **quality-based tile selection outperforms random selection**, proving the value of Stage 1 in the pipeline. Key metrics to compare:

- **Accuracy** (tile and WSI level)
- **Balanced Accuracy**
- **F1 Score** (macro)
- **Matthews Correlation Coefficient**
- **Cohen's Kappa**
- **AUC** (micro and macro)

## Troubleshooting

### Issue: Not enough tiles
If a WSI has fewer than 200 tiles, the script will select all available tiles and print a warning.

### Issue: Storage full
Use the `--no_cleanup` flag cautiously. The default behavior removes unselected tiles to save space.

### Issue: Missing cohort JSON
Ensure the cohort JSON file exists and follows the format:
```json
{
  "training": {
    "ALL": ["TV0001", "TV0002", ...],
    "AML_APL": [...],
    ...
  },
  "validation": {...},
  "test": {...}
}
```

## Notes

- Random selection uses Python's `random.sample()` for uniform random sampling
- The same excluded tiles (TV0577 specific tiles) are handled as in the original pipeline
- All scripts support command-line help: `python script.py --help`
