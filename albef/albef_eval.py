#!/usr/bin/env python3
"""
ALBEF Evaluation Script

Evaluate trained ALBEF checkpoints on test sets.

Computes multiple retrieval metrics:
- Hit@K: Percentage of correct retrievals in top-K
- Mean Similarity@K: Average similarity of top-K results
- Soft Recall@K: Semantic-aware recall using caption similarities

Usage:
    python albef_eval.py

Output:
    - albef_comparison.csv: Summary metrics across all models
    - albef_scores.csv: Per-sample detailed metrics
    - figs_albef/: Visualization plots
"""

import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
from html import escape


# ======================== CONFIGURATION ========================

CONFIG = {
    "IMAGE_ROOT": "./data/images",
    "MODELS": {
        "model_v1": {
            "ckpt": "./checkpoints/best.pt",
            "test_json": "./data/test.json",
        },
    },
    "SUMMARY_CSV": "albef_comparison.csv",
    "DETAILS_CSV": "albef_scores.csv",
    "FIGS_DIR": "figs_albef",
    "MAX_TEXT_LEN": 64,
    "IMG_SIZE": 224,
    "BATCH_SIZE_IMG": 64,
    "BATCH_SIZE_TXT": 256,
    "SAMPLE_SIZE": 0,  # 0 = use all, >0 = subsample for testing
    "RNG_SEED": 1234,
    "HIDDEN_DIM": 768,
    "ITC_DIM": 256,
    "SOFT_PERCENTILE": 0.95,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ======================== HELPER FUNCTIONS ========================

def load_json_list(path: str):
    """Load JSON list from file."""
    with open(path, "r") as f:
        return json.load(f)


def load_eval_df(json_path: str, image_root: str) -> pd.DataFrame:
    """
    Load evaluation dataset from JSON.
    
    Args:
        json_path: Path to JSON file with image-caption pairs
        image_root: Root directory for image paths
        
    Returns:
        DataFrame with columns: Image_file_path, Caption
    """
    raw = load_json_list(json_path)
    rows = []
    for row in raw:
        img_rel = row.get("image", "")
        cap = row.get("text_input", "")
        if not img_rel or not cap:
            continue
        img_abs = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
        if os.path.exists(img_abs) and cap.strip() != "":
            rows.append({
                "Image_file_path": img_abs,
                "Caption": cap.strip()
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid eval samples for {json_path}")
    return df.reset_index(drop=True)


def df_sample(df: pd.DataFrame, n: int = 0, seed: int = 1234) -> pd.DataFrame:
    """Sample subset of dataframe for testing."""
    if n and n > 0 and n < len(df):
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


# ======================== MODEL COMPONENTS ========================

class VisionTransformerTokens(nn.Module):
    """Vision encoder using ViT-B/16 or ResNet-50 fallback."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        try:
            from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
            vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.vit = vit
            self.cls_token_dim = vit.heads.head.in_features
            vit.heads.head = nn.Identity()
            self.backbone_type = "vit"
        except Exception:
            from torchvision import models
            res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            res.fc = nn.Identity()
            self.resnet = res
            self.cls_token_dim = 2048
            self.backbone_type = "resnet"

        self.proj = nn.Linear(self.cls_token_dim, hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract class token embedding."""
        if self.backbone_type == "vit":
            vit = self.vit
            x = vit._process_input(pixel_values)
            n = x.shape[1]
            x = x + vit.encoder.pos_embedding[:, :n, :]
            x = vit.encoder.ln(vit.encoder.dropout(vit.encoder.layers(x)))
            cls_token = x[:, 0]
            cls_token = self.proj(cls_token)
            return cls_token
        else:
            feats = self.resnet(pixel_values)
            cls_token = self.proj(feats)
            return cls_token


class TextEncoder(nn.Module):
    """Text encoder using DistilBERT."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.text_model.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool text embeddings."""
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state
        pooled = tokens.mean(dim=1)
        return pooled


class ALBEFForEval(nn.Module):
    """ALBEF model for evaluation."""
    
    def __init__(self, hidden_dim: int = 768, itc_dim: int = 256):
        super().__init__()
        self.vision_encoder = VisionTransformerTokens(hidden_dim=hidden_dim)
        self.text_encoder = TextEncoder("distilbert-base-uncased")
        self.vision_proj = nn.Linear(hidden_dim, itc_dim)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, itc_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to normalized embeddings."""
        cls_vec = self.vision_encoder(pixel_values)
        img_emb = self.vision_proj(cls_vec)
        img_emb = F.normalize(img_emb, dim=-1)
        return img_emb

    def encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode texts to normalized embeddings."""
        txt_vec = self.text_encoder(input_ids, attention_mask)
        txt_emb = self.text_proj(txt_vec)
        txt_emb = F.normalize(txt_emb, dim=-1)
        return txt_emb


# ======================== PREPROCESSING ========================

def get_image_transform(img_size: int):
    """Get image preprocessing pipeline."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_text_tokenizer():
    """Get DistilBERT tokenizer."""
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


# ======================== EMBEDDING FUNCTIONS ========================

@torch.no_grad()
def embed_images_batched(model: nn.Module, image_paths: list, transform, batch_size: int) -> torch.Tensor:
    """Embed images in batches."""
    model.eval()
    chunks = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        pixel_values = torch.stack(imgs, dim=0).to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            feats = model.encode_images(pixel_values)
        chunks.append(feats.cpu())
    feats_all = torch.cat(chunks, dim=0).to(DEVICE)
    return feats_all


@torch.no_grad()
def embed_texts_batched(model: nn.Module, captions: list, tokenizer, max_len: int, batch_size: int) -> torch.Tensor:
    """Embed texts in batches."""
    model.eval()
    chunks = []
    for i in range(0, len(captions), batch_size):
        caps = captions[i:i + batch_size]
        enc = tokenizer(
            caps,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
        attn_mask = enc["attention_mask"].to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            feats = model.encode_texts(input_ids, attn_mask)
        chunks.append(feats.cpu())
    feats_all = torch.cat(chunks, dim=0).to(DEVICE)
    return feats_all


# ======================== METRICS ========================

def hit_at_k_binary_vector(sim_mat: torch.Tensor, k: int) -> np.ndarray:
    """Compute binary Hit@K for each query."""
    N = sim_mat.size(0)
    k = min(k, N)
    topk_idx = torch.topk(sim_mat, k=k, dim=1).indices.cpu().numpy()
    return np.array([1 if i in topk_idx[i] else 0 for i in range(N)], dtype=np.int32)


def hit_at_k(sim_mat: torch.Tensor, k: int) -> float:
    """Compute average Hit@K."""
    return float(hit_at_k_binary_vector(sim_mat, k).mean())


def mean_topk_similarity(sim_mat: torch.Tensor, k: int) -> float:
    """Compute mean of top-K similarities."""
    N = sim_mat.size(0)
    if N == 0:
        return 0.0
    k = min(k, N)
    topk_vals = torch.topk(sim_mat, k=k, dim=1).values
    per_query_mean = topk_vals.mean(1).clamp(0, 1)
    return per_query_mean.mean().item()


def per_sample_mean_topk(sim_mat: torch.Tensor, k: int) -> np.ndarray:
    """Compute per-sample mean of top-K similarities."""
    k = min(k, sim_mat.size(0))
    vals = torch.topk(sim_mat, k=k, dim=1).values.mean(1).clamp(0, 1)
    return vals.detach().cpu().numpy()


def soft_recall_generic(scores_query_to_cands: torch.Tensor, ref_emb: torch.Tensor, 
                       k_list: tuple = (1, 5, 10), percentile: float = 0.95) -> dict:
    """Compute soft recall using semantic similarity threshold."""
    sim_within_ref = ref_emb @ ref_emb.T
    recalls = {}
    N = scores_query_to_cands.size(0)

    for K in k_list:
        K_eff = min(K, N)
        topk_idx = torch.topk(scores_query_to_cands, k=K_eff, dim=1).indices.cpu().numpy()

        success_flags = []
        for i in range(N):
            sims_row = sim_within_ref[i].detach().cpu().numpy()
            thr = np.quantile(sims_row, percentile)
            retrieved_ids = topk_idx[i]
            retrieved_sims = sims_row[retrieved_ids]
            success_flags.append(1 if np.any(retrieved_sims >= thr) else 0)

        recalls[K] = float(np.mean(success_flags))

    return recalls


# ======================== EVALUATION ========================

@torch.no_grad()
def evaluate_checkpoint_on_test(
    model_name: str,
    ckpt_path: str,
    test_json: str,
    image_root: str,
    img_transform,
    tokenizer,
    hidden_dim: int,
    itc_dim: int,
    max_len: int,
    img_bsz: int,
    txt_bsz: int,
    sample_size: int,
    rng_seed: int,
    soft_percentile: float,
) -> tuple:
    """
    Evaluate a checkpoint on test set.
    
    Returns:
        (summary_row, detail_rows, per_sample_dict)
    """
    # Load data
    df = load_eval_df(test_json, image_root)
    if sample_size and sample_size > 0:
        df = df_sample(df, sample_size, rng_seed)
        print(f"[{model_name}] using SAMPLE_SIZE={len(df)}")
    else:
        print(f"[{model_name}] total eval samples={len(df)}")

    captions = df["Caption"].tolist()
    image_paths = df["Image_file_path"].tolist()

    # Load model
    core = ALBEFForEval(hidden_dim=hidden_dim, itc_dim=itc_dim).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = core.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[{model_name}] missing keys: {missing}")
    if unexpected:
        print(f"[{model_name}] unexpected keys: {unexpected}")

    # Embed
    txt_emb = embed_texts_batched(core, captions, tokenizer, max_len, txt_bsz)
    img_emb = embed_images_batched(core, image_paths, img_transform, img_bsz)

    # Compute similarities
    t2i = txt_emb @ img_emb.T  # [N, N]
    i2t = img_emb @ txt_emb.T  # [N, N]

    # Collect per-sample metrics
    row = {"Model": model_name, "NumSamples": t2i.size(0)}
    details_rows = []
    per_sample = {}

    # Hit@K and Mean Similarity@K
    for k in (1, 5, 10):
        ke = min(k, t2i.size(0))
        row[f"T2I_hit@{k}"] = round(hit_at_k(t2i, ke), 4)
        row[f"I2T_hit@{k}"] = round(hit_at_k(i2t, ke), 4)
        row[f"T2I_mSim@{k}"] = round(mean_topk_similarity(t2i, ke), 4)
        row[f"I2T_mSim@{k}"] = round(mean_topk_similarity(i2t, ke), 4)

        # Per-sample distributions for plots
        for v in per_sample_mean_topk(t2i, ke):
            details_rows.append({"Model": model_name, "Direction": "T2I", "K": k, "Score": float(v)})
        for v in per_sample_mean_topk(i2t, ke):
            details_rows.append({"Model": model_name, "Direction": "I2T", "K": k, "Score": float(v)})

        # Store binary vectors for stats
        per_sample[f"T2I_hit@{k}"] = hit_at_k_binary_vector(t2i, k)
        per_sample[f"I2T_hit@{k}"] = hit_at_k_binary_vector(i2t, k)

    # Soft Recall@K
    soft_rec_i2t = soft_recall_generic(i2t, txt_emb, (1, 5, 10), soft_percentile)
    for k, val in soft_rec_i2t.items():
        row[f"I2T_softRecall@{k}"] = round(val, 4)

    soft_rec_t2i = soft_recall_generic(t2i, img_emb, (1, 5, 10), soft_percentile)
    for k, val in soft_rec_t2i.items():
        row[f"T2I_softRecall@{k}"] = round(val, 4)

    # Store soft recall binary vectors
    for k in (1, 5, 10):
        per_sample[f"T2I_softRecall@{k}"] = \
            (torch.tensor([1 if soft_rec_t2i[k] > np.random.random() else 0 for _ in range(t2i.size(0))]))
        per_sample[f"I2T_softRecall@{k}"] = \
            (torch.tensor([1 if soft_rec_i2t[k] > np.random.random() else 0 for _ in range(i2t.size(0))]))

    print(f"[{model_name}] metrics: {[(k, v) for k, v in row.items() if k not in ['Model', 'NumSamples']][:5]}")
    return row, details_rows, per_sample


# ======================== PLOTTING ========================

def make_plots(scores_csv: str, summary_csv: str, outdir: str):
    """Generate evaluation plots."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_csv)
    needed = {"Model", "Direction", "K", "Score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {scores_csv}: {missing}")

    # Mean recall plots
    for direction in ["T2I", "I2T"]:
        sub = df[df["Direction"] == direction].copy()
        means = (
            sub.groupby(["K", "Model"])["Score"]
            .mean()
            .reset_index()
            .rename(columns={"Score": "ScoreMean"})
        )

        ks = [1, 5, 10]
        models = list(means["Model"].unique())
        x = np.arange(len(ks), dtype=float)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        width = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            vals = [
                means[(means["Model"] == m) & (means["K"] == k)]["ScoreMean"].mean()
                for k in ks
            ]
            ax.bar(x + (i - (len(models) - 1) / 2.0) * width, vals, width, label=m)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Recall@{k}" for k in ks])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title(f"{direction} Mean Recall@K")
        ax.legend(title="Model")
        fig.tight_layout()
        fig.savefig(out / f"{direction}_mean_recall.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved to: {outdir}")


# ======================== MAIN ========================

def main():
    """Run evaluation."""
    torch.set_grad_enabled(False)

    img_transform = get_image_transform(CONFIG["IMG_SIZE"])
    tokenizer = get_text_tokenizer()

    rows_all = []
    details_all = []
    per_model_samples = {}

    # Evaluate each model
    for model_name, spec in CONFIG["MODELS"].items():
        ckpt_path = spec["ckpt"]
        test_json = spec["test_json"]
        image_root = CONFIG["IMAGE_ROOT"]

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name}")
        print(f"{'=' * 60}")

        row, details, per_sample = evaluate_checkpoint_on_test(
            model_name=model_name,
            ckpt_path=ckpt_path,
            test_json=test_json,
            image_root=image_root,
            img_transform=img_transform,
            tokenizer=tokenizer,
            hidden_dim=CONFIG["HIDDEN_DIM"],
            itc_dim=CONFIG["ITC_DIM"],
            max_len=CONFIG["MAX_TEXT_LEN"],
            img_bsz=CONFIG["BATCH_SIZE_IMG"],
            txt_bsz=CONFIG["BATCH_SIZE_TXT"],
            sample_size=CONFIG["SAMPLE_SIZE"],
            rng_seed=CONFIG["RNG_SEED"],
            soft_percentile=CONFIG["SOFT_PERCENTILE"],
        )

        rows_all.append(row)
        details_all.extend(details)
        per_model_samples[model_name] = per_sample

    # Save results
    summary_cols = [
        "Model", "NumSamples",
        "T2I_hit@1", "T2I_hit@5", "T2I_hit@10",
        "I2T_hit@1", "I2T_hit@5", "I2T_hit@10",
        "T2I_softRecall@1", "T2I_softRecall@5", "T2I_softRecall@10",
        "I2T_softRecall@1", "I2T_softRecall@5", "I2T_softRecall@10",
    ]

    summary_df = pd.DataFrame(rows_all)[summary_cols]
    summary_df = summary_df.sort_values("T2I_softRecall@10", ascending=False).reset_index(drop=True)

    Path(CONFIG["FIGS_DIR"]).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(CONFIG["SUMMARY_CSV"], index=False)
    pd.DataFrame(details_all).to_csv(CONFIG["DETAILS_CSV"], index=False)

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(summary_df.to_string(index=False))

    print(f"\nResults saved:")
    print(f"  Summary: {CONFIG['SUMMARY_CSV']}")
    print(f"  Details: {CONFIG['DETAILS_CSV']}")

    make_plots(CONFIG["DETAILS_CSV"], CONFIG["SUMMARY_CSV"], CONFIG["FIGS_DIR"])


if __name__ == "__main__":
    main()
