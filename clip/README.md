# CLIP Models

This folder contains CLIP-based models for image-caption retrieval on medical data.

## Files

- **baseline2.py** - Trains a CLIP model with attention mechanism on image-caption pairs
- **qual_eval_clip.py** - Qualitative evaluation: retrieves top-1 image-caption matches and saves results to CSV
- **recall_clip.py** - Quantitative evaluation: computes soft recall@k metrics across baselines

## Usage

### Training
```bash
python baseline2.py
```

### Evaluation
```bash
python recall_clip.py          
python qual_eval_clip.py       
```


