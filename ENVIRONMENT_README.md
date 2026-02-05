# BM_cytology Environment Files

This directory contains two environment specification files:

## 1. `environment.yml` (Recommended)
**Minimal, cross-platform compatible specification**

Contains only the essential packages you explicitly need. This is the **recommended file** for:
- Creating the environment on different machines
- Sharing with collaborators
- Version control

**Usage:**
```bash
conda env create -f environment.yml
```

**Post-installation fix required:**
After creating the environment, you need to patch the `roc-utils` package for scipy 1.15+ compatibility:
```bash
conda activate BM_cytology

# Find and edit the _roc.py file
ROCS_FILE=$(python -c "import roc_utils, os; print(os.path.join(os.path.dirname(roc_utils.__file__), '_roc.py'))")

# Replace 'from scipy import interp' with 'interp = np.interp'
sed -i 's/from scipy import interp/# scipy.interp was deprecated and removed in scipy 1.x, use numpy.interp instead\ninterp = np.interp/' "$ROCS_FILE"
```

**Important:** You also need to create the activation script for CUDA library paths:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << 'EOF'
#!/bin/bash
# Set LD_LIBRARY_PATH to include CUDA libraries from conda environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

## 2. `environment_full.yml`
**Complete specification with all dependencies and exact versions**

Contains every package with exact versions. Use this if:
- You need to reproduce the exact environment
- You're on the same platform (Linux x86_64)

**Usage:**
```bash
conda env create -f environment_full.yml
```

**Note:** This file includes platform-specific packages and may not work on different operating systems or architectures.

## Key Packages

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.0 | Runtime |
| TensorFlow | 2.10.0 | Deep learning framework |
| CUDA Toolkit | 11.8 | GPU acceleration |
| cuDNN | 8.8 | Neural network primitives |
| roc-utils | 0.2.2 | ROC curve bootstrap analysis |
| OpenSlide | 4.0.0 | WSI processing |

## Verification

After creating the environment, verify GPU support:
```bash
conda activate BM_cytology
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

Expected output: `GPUs: 1` (or more, depending on your system)
