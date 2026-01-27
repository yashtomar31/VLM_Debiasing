# ALBEF: Align Before Fusing for Vision-Language Retrieval

This repository contains an implementation of ALBEF (Align Before Fusing) for medical image-text retrieval on the PMC-OA (PubMed Central Open Access) dataset. The model combines image-text contrastive learning (ITC) with image-text matching (ITM) to learn aligned vision-language representations.

## Overview

**ALBEF** is a vision-language model that learns to:
1. **Align** image and text embeddings through contrastive learning (ITC head)
2. **Fuse** aligned representations to determine if pairs match (ITM head)

Key features:
- ✅ Supports Vision Transformer (ViT-B/16) or ResNet-50 as vision backbone
- ✅ Uses DistilBERT for text encoding
- ✅ Multi-GPU training with DistributedDataParallel
- ✅ Mixed precision training (AMP)
- ✅ Comprehensive evaluation metrics: Hit@K, Mean Similarity, Soft Recall
- ✅ Statistical significance testing and HTML comparison reports

## Project Structure

```
albef/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── download_common_pmcoa.py          # Download PMC-OA dataset
├── albef_train.py                    # Training script
├── albef_test.py                     # Evaluation script
└── retrieve_html_011926.py           # Evaluation with HTML comparison
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 200+ GB disk space (for full PMC-OA dataset)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd albef
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

### Step 1: Download PMC-OA Dataset

The PMC-OA dataset contains medical paper figures. Download it using:

```bash
python download_common_pmcoa.py
```

**Configuration** (`download_common_pmcoa.py`):
- `BASE_URL`: NCBI FTP server URL
- `DATA_FILE`: CSV file with metadata (must contain `Online_file_path` column)
- `DOWNLOAD_DIR`: Output directory for extracted images

**Requirements**:
- A CSV metadata file at `./csv/VLM_train.csv` with column `Online_file_path`
- Internet connection for NCBI FTP downloads

### Step 2: Prepare JSON Files

Create JSON files with image-caption pairs in this format:

```json
[
  {
    "image": "path/to/image.jpg",
    "text_input": "Description of the image"
  },
  ...
]
```

Create three JSON files:
- `train.json` - Training data
- `val.json` - Validation data  
- `test.json` - Test data

## Usage

### Training

Edit configuration in `albef_train.py`:

```python
TRAIN_JSON      = "/path/to/train.json"
VAL_JSON        = "/path/to/val.json"
IMAGE_ROOT      = "/path/to/images/"

BATCH_SIZE      = 128          # Tune based on GPU memory
EPOCHS          = 30
LR              = 1e-4
SAVE_DIR        = "./checkpoints"
```

**Single GPU training**:
```bash
python albef_train.py
```

**Multi-GPU training** (DistributedDataParallel):
```bash
torchrun --nproc_per_node=4 albef_train.py
```

**Output**:
- Checkpoints saved to `SAVE_DIR/`
- Best model: `SAVE_DIR/best.pt`
- Periodic: `SAVE_DIR/step{step}_epoch{epoch}.pt`

### Evaluation

#### Option 1: Basic Evaluation

Edit configuration in `albef_test.py`:

```python
CONFIG = {
    "IMAGE_ROOT": "/path/to/images",
    "MODELS": {
        "model_name": {
            "ckpt": "./checkpoints/best.pt",
            "test_json": "/path/to/test.json",
        },
    },
    ...
}
```

Run evaluation:
```bash
python albef_test.py
```

#### Option 2: Evaluation with Comparison HTML

Use `retrieve_html_011926.py` for detailed analysis:

```bash
python retrieve_html_011926.py
```

**Output files**:
- `albef_comparison.csv` - Summary metrics for all models
- `albef_scores.csv` - Per-sample metrics
- `figs_albef/` - Visualization plots (bar charts, box plots)
- `albef_model3_wins.html` - Interactive HTML comparison (if model3 outperforms model1/2)

## Evaluation Metrics

The model is evaluated using several retrieval metrics:

### Strict Metrics
- **Hit@K**: Percentage of queries where ground truth is in top-K results
- **Mean Similarity@K**: Average similarity of top-K retrieved items

### Semantic Metrics  
- **Soft Recall@K**: Uses semantic similarity within caption space to determine "soft" correctness
  - Threshold: 95th percentile similarity between captions
  - Success if any top-K item is semantically similar to GT

### Statistics
- McNemar's test for comparing binary metrics (p-values & significance stars)
- Paired t-test for continuous metrics

## Configuration Reference

### Training Config (`albef_train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | 224 | Input image resolution |
| `MAX_TEXT_LEN` | 64 | Max caption token length |
| `BATCH_SIZE` | 128 | Batch size (tune for GPU memory) |
| `EPOCHS` | 30 | Number of training epochs |
| `LR` | 1e-4 | Learning rate |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay |
| `WARMUP_STEPS` | 1000 | Linear warmup steps |
| `HIDDEN_DIM` | 768 | Transformer hidden dimension |
| `ITC_DIM` | 256 | Projection dimension for contrastive head |

### Evaluation Config (`albef_test.py`, `retrieve_html_011926.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_TEXT_LEN` | 64 | Max caption token length |
| `IMG_SIZE` | 224 | Image resolution for evaluation |
| `BATCH_SIZE_IMG` | 64 | Image embedding batch size |
| `BATCH_SIZE_TXT` | 256 | Text embedding batch size |
| `SAMPLE_SIZE` | 0 | Sample size (0 = full test set) |
| `SOFT_PERCENTILE` | 0.95 | Semantic similarity threshold for soft recall |

## Model Architecture

### Vision Encoder
- **Backbone**: Vision Transformer (ViT-B/16) or ResNet-50
- **Output**: Class token + patch embeddings
- **Projection**: Linear layer to `HIDDEN_DIM` (768)

### Text Encoder
- **Backbone**: DistilBERT (768 hidden size)
- **Output**: Mean-pooled token embeddings
- **Projection**: Linear layer to `ITC_DIM` (256) for contrastive head

### Fusion Encoder (ITM)
- **Architecture**: 2-layer Transformer
- **Purpose**: Fuse text & image tokens to predict match/mismatch
- **Output**: 2-class logits (match vs mismatch)

### Learning Objectives

1. **ITC Loss** (Contrastive):
   - Align global image & text embeddings
   - InfoNCE loss: maximize similarity of matched pairs, minimize others
   - Symmetric: loss_i + loss_t

2. **ITM Loss** (Classification):
   - Determine if image-text pair is real or shuffled negative
   - Binary cross-entropy on fused representations
   - In-batch negatives: shuffle text within batch

## Outputs

### Training Outputs
```
./checkpoints/
├── best.pt                          # Best model by validation loss
└── step{step}_epoch{epoch}.pt       # Periodic checkpoints
```

### Evaluation Outputs
```
./albef_comparison.csv               # Summary metrics table
./albef_scores.csv                   # Per-sample detailed metrics
./figs_albef/
├── T2I_mean_recall_at_k.png        # Text-to-Image retrieval plot
├── I2T_mean_recall_at_k.png        # Image-to-Text retrieval plot
├── T2I_recall_at_k_boxplot.png     # Distribution plots
└── I2T_recall_at_k_boxplot.png
./albef_model3_wins.html            # Interactive comparison (when applicable)
```

### CSV Column Reference

**albef_comparison.csv**:
- `Model`: Model name
- `NumSamples`: Number of test samples
- `T2I_hit@{1,5,10}`: Text-to-Image Hit@K
- `I2T_hit@{1,5,10}`: Image-to-Text Hit@K
- `T2I_softRecall@{1,5,10}`: Semantic soft recall (T→I)
- `I2T_softRecall@{1,5,10}`: Semantic soft recall (I→T)
- `*_p_vs_baseline`: p-values from McNemar's test
- `*_starred`: Metric with significance stars (*, **, ***)

## Example Workflow

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download and prepare data
python download_common_pmcoa.py
# Create train.json, val.json, test.json with image-caption pairs

# 3. Train model
python albef_train.py

# 4. Evaluate single model
python albef_test.py

# 5. Evaluate with comparison
python retrieve_html_011926.py
# Check outputs: albef_comparison.csv, figs_albef/, albef_model3_wins.html
```

## Troubleshooting

### OOM (Out of Memory)
- Reduce `BATCH_SIZE` in training config
- Reduce `IMG_SIZE` (e.g., 192 instead of 224)
- Use gradient accumulation (add to training loop)

### Missing checkpoint
- Verify `SAVE_DIR` directory exists
- Check file permissions
- Ensure sufficient disk space

### Evaluation errors
- Verify JSON files have `"image"` and `"text_input"` keys
- Check image paths exist (absolute or relative to `IMAGE_ROOT`)
- Ensure checkpoint file exists at specified path

### CUDA memory issues
- Use smaller batch sizes
- Enable mixed precision (already enabled by default)
- Consider single-GPU training first

## References

The implementation is based on:
- **ALBEF**: Align Before Fusing - Learning Unified Representations of Medical Images and Text
- **ViT**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **DistilBERT**: Sanh et al., "DistilBERT, a distilled version of BERT"
- **CLIP**: Radford et al., "Learning Transferable Models for Image-Text Retrieval"


## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
