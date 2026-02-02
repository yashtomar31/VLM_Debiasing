# VLM Debiasing

This project contains two vision-language models for medical image-caption retrieval: CLIP and ALBEF.

## Project Structure

### `/clip`
CLIP-based models for image-caption retrieval with attention mechanisms.

**Files:**
- `baseline2.py` - Train CLIP model with attention on image-caption pairs
- `qual_eval_clip.py` - Qualitative evaluation: top-1 retrieval results
- `recall_clip.py` - Quantitative evaluation: recall@k metrics

**Usage:**
```bash
cd clip
python baseline2.py              # Train
python recall_clip.py             # Evaluate metrics
python qual_eval_clip.py          # Qualitative results
```

### `/albef`
ALBEF (Align Before Fusing) for vision-language retrieval with ITC + ITM heads.

**Files:**
- `albef_train.py` - Training script
- `albef_test.py` - Evaluation script
- `retrieve_html.py` - HTML comparison reports
- `download_common_pmcoa.py` - Download PMC-OA dataset

**Usage:**
```bash
cd albef
python albef_train.py            # Train
python albef_test.py             # Evaluate
python retrieve_html.py          # Generate reports
```

## Requirements

Both models require:
- torch
- torchvision
- transformers
- pandas
- PIL
- scikit-learn
- matplotlib

See individual folders for detailed documentation.
