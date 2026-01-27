#!/usr/bin/env python3
"""
ALBEF: Align Before Fusing for Vision-Language Retrieval

This script trains an ALBEF model combining:
- ITC (Image-Text Contrastive) head for alignment
- ITM (Image-Text Matching) head for fusion

Supports single-GPU and multi-GPU distributed training.

Usage:
    # Single GPU
    python albef_train.py
    
    # Multi GPU (4 GPUs)
    torchrun --nproc_per_node=4 albef_train.py
"""

import os
import json
import math
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

# =============================================================================
# CONFIG
# =============================================================================

TRAIN_JSON      = "/home/yash/train_model2.json"
VAL_JSON        = "/home/yash/val_model2.json"
IMAGE_ROOT      = "/home/yash/common_pmcoa_data"

# TRAIN_JSON      = "/home/yash/train_model3.json"
# VAL_JSON        = "/home/yash/val_model3.json"
# IMAGE_ROOT      = "/home/yash/data"

TEXT_COL        = "text_input"
IMG_COL         = "image"

IMAGE_SIZE      = 224
MAX_TEXT_LEN    = 64

BATCH_SIZE      = 128           # tune for VRAM
EPOCHS          = 30
ITC_DIM         = 256           # projection dim for contrastive head
HIDDEN_DIM      = 768           # transformer width, must match DistilBERT hidden size
LR              = 1e-4
WEIGHT_DECAY    = 0.01
WARMUP_STEPS    = 1000
SAVE_DIR        = "./checkpoints_albef_model2"
os.makedirs(SAVE_DIR, exist_ok=True)

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK       = int(os.environ.get("RANK", 0))
DEVICE     = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DATA
# =============================================================================

def load_json_list(path):
    # expects a list[dict] JSON array
    # if you actually have jsonlines (one sample per line), replace with a line loop
    with open(path, "r") as f:
        return json.load(f)

class ImageTextDataset(Dataset):
    def __init__(self, json_path, image_root, tokenizer, transform,
                 text_key=TEXT_COL, img_key=IMG_COL, max_text_len=MAX_TEXT_LEN):
        raw = load_json_list(json_path)
        samples = []
        for row in raw:
            img_rel = row.get(img_key, None)
            txt     = row.get(text_key, None)
            if not img_rel or not txt:
                continue
            full_path = os.path.join(image_root, img_rel)
            if os.path.exists(full_path):
                samples.append({
                    "image_path": full_path,
                    "text": txt
                })

        self.samples = samples
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]

        # image -> tensor
        image = Image.open(ex["image_path"]).convert("RGB")
        pixel_values = self.transform(image)

        # text -> token ids
        enc = self.tokenizer(
            ex["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        )
        input_ids      = enc["input_ids"].squeeze(0)          # [L]
        attention_mask = enc["attention_mask"].squeeze(0)     # [L]

        return pixel_values, input_ids, attention_mask


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# =============================================================================
# MODEL BACKBONES
# =============================================================================

class VisionTransformerTokens(nn.Module):
    """
    ViT-B/16 style vision encoder that returns:
    - patch+cls tokens
    - global CLS-like embedding
    Falls back to ResNet50 global if torchvision ViT is unavailable.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        try:
            from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
            vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.vit = vit
            self.cls_token_dim = vit.heads.head.in_features  # typically 768
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

    def forward(self, pixel_values):
        """
        Returns:
            vision_tokens: [B, N_img, HIDDEN_DIM]
            vision_cls:    [B, HIDDEN_DIM]
        """
        if self.backbone_type == "vit":
            vit = self.vit
            # patch embed + class token
            x = vit._process_input(pixel_values)               # [B, 1+Npatch, dim]
            n = x.shape[1]

            # positional embeddings
            x = x + vit.encoder.pos_embedding[:, :n, :]

            # transformer encoder forward
            x = vit.encoder.ln(vit.encoder.dropout(vit.encoder.layers(x)))
            # x: [B, 1+Npatch, dim]

            cls_token = x[:,0]                 # [B, dim]
            cls_token = self.proj(cls_token)   # [B, HIDDEN_DIM]

            tokens = self.proj(x)              # [B, 1+Npatch, HIDDEN_DIM]
            return tokens, cls_token

        else:
            # ResNet fallback: produce only global embedding
            feats = self.resnet(pixel_values)         # [B,2048]
            cls_token = self.proj(feats)              # [B,HIDDEN_DIM]
            tokens = cls_token.unsqueeze(1)           # [B,1,HIDDEN_DIM]
            return tokens, cls_token


class TextEncoder(nn.Module):
    """
    DistilBERT text tower.
    Outputs:
    - text_tokens: [B, Nt, Htxt]
    - text_cls:    [B, Htxt]  (mean pooled)
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.text_model.config.hidden_size  # 768 for distilbert-base-uncased

    def forward(self, input_ids, attention_mask):
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state                   # [B,Nt,Htxt]
        cls_like = tokens.mean(dim=1)                    # [B,Htxt]
        return tokens, cls_like


class FusionEncoder(nn.Module):
    """
    Small Transformer that takes [text tokens ; image CLS token] and lets them attend.
    We then classify match vs mismatch using fused [CLS]-like output.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, num_layers=2, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, joint_tokens, attn_mask):
        """
        joint_tokens: [B, Nt+1, H]
        attn_mask:    [B, Nt+1]  (1=keep, 0=pad)
        """
        pad_mask = (attn_mask == 0)  # bool [B, Nt+1], True where we should IGNORE
        fused = self.encoder(joint_tokens, src_key_padding_mask=pad_mask)  # [B,Nt+1,H]
        fused_cls = fused[:,0]  # take first text token position as pooled representation
        return fused_cls


class ALBEFModel(nn.Module):
    """
    ALBEF-style:
    - ITC head (contrastive alignment of global embeddings)
    - ITM head (match / mismatch via fusion encoder)
    """
    def __init__(self, itc_dim=ITC_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.vision_encoder = VisionTransformerTokens(hidden_dim=hidden_dim)
        self.text_encoder   = TextEncoder("distilbert-base-uncased")
        self.fusion_encoder = FusionEncoder(hidden_dim=hidden_dim)

        # projection heads for ITC (contrastive)
        self.vision_proj = nn.Linear(hidden_dim, itc_dim)
        self.text_proj   = nn.Linear(self.text_encoder.hidden_size, itc_dim)

        # ITM head: binary (match / mismatch)
        self.itm_head = nn.Linear(hidden_dim, 2)

        # learnable temperature for ITC logits
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))

    # ---------- ITC branch ----------
    def get_global_embeddings_and_logits(self, pixel_values, input_ids, attention_mask):
        """
        ITC: produce global embeddings for image and text,
        compute pairwise similarity matrix across the batch.
        """
        vision_tokens, vision_cls = self.vision_encoder(pixel_values)         # [B,Nv,H], [B,H]
        text_tokens, text_cls     = self.text_encoder(input_ids, attention_mask)  # [B,Nt,Htxt]

        img_emb = F.normalize(self.vision_proj(vision_cls), dim=-1)           # [B,itc_dim]
        txt_emb = F.normalize(self.text_proj(text_cls),   dim=-1)             # [B,itc_dim]

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_emb @ txt_emb.t()                          # [B,B]

        return img_emb, txt_emb, logits

    # ---------- ITM branch ----------
    def get_itm_logits(self, pixel_values, input_ids, attention_mask):
        """
        Build joint tokens = [text tokens ; vision CLS token]
        Then classify whether the pair is a real pair or a shuffled negative.
        """
        vision_tokens, vision_cls = self.vision_encoder(pixel_values)         # [B,Nv,H],[B,H]
        text_tokens, _            = self.text_encoder(input_ids, attention_mask)  # [B,Nt,Htxt]

        # We assume HIDDEN_DIM == text hidden size == vision hidden size (768).
        assert text_tokens.size(-1) == vision_tokens.size(-1), \
            "text hidden dim must equal vision hidden dim (HIDDEN_DIM should be 768)."

        vision_cls_token = vision_cls.unsqueeze(1)                            # [B,1,H]
        joint_tokens = torch.cat([text_tokens, vision_cls_token], dim=1)      # [B,Nt+1,H]

        # Build mask: text attention_mask + 1 for the image token
        B, Nt, _ = text_tokens.shape
        vision_mask = torch.ones((B,1), device=attention_mask.device, dtype=attention_mask.dtype)
        joint_mask  = torch.cat([attention_mask, vision_mask], dim=1)         # [B,Nt+1]

        fused_cls = self.fusion_encoder(joint_tokens, joint_mask)             # [B,H]
        logits_itm = self.itm_head(fused_cls)                                 # [B,2]
        return logits_itm


# =============================================================================
# LOSSES
# =============================================================================

def itc_loss_from_logits(logits):
    """
    Image-Text Contrastive (InfoNCE-style, like CLIP/ALBEF).
    logits: [B,B] where logits[i,j] = similarity(img_i, text_j)
    We want diagonal (i,i) to be largest.
    """
    bsz = logits.size(0)
    labels = torch.arange(bsz, device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)

def build_itm_batch(model, pixel_values, input_ids, attention_mask):
    """
    Create positives and in-batch negatives for ITM:
    positives = (img_i, text_i)
    negatives = (img_i, text_j≠i)
    """
    B = pixel_values.size(0)

    # positives
    logits_pos = model.get_itm_logits(pixel_values, input_ids, attention_mask)  # [B,2]

    # negatives: shuffle text
    shuffle_idx = torch.randperm(B, device=pixel_values.device)
    neg_input_ids      = input_ids[shuffle_idx]
    neg_attention_mask = attention_mask[shuffle_idx]
    logits_neg = model.get_itm_logits(pixel_values, neg_input_ids, neg_attention_mask)  # [B,2]

    logits_all = torch.cat([logits_pos, logits_neg], dim=0)  # [2B,2]
    labels_all = torch.cat([
        torch.ones(B, dtype=torch.long, device=pixel_values.device),   # 1 = match
        torch.zeros(B, dtype=torch.long, device=pixel_values.device)   # 0 = mismatch
    ], dim=0)  # [2B]

    return logits_all, labels_all

def itm_loss_from_logits(logits_all, labels_all):
    return F.cross_entropy(logits_all, labels_all)


# =============================================================================
# LR SCHEDULE
# =============================================================================

def cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# =============================================================================
# TRAIN / EVAL LOOPS
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, epoch, global_step, total_steps, scaler):
    model.train()

    sampler = dataloader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)

    running_loss = 0.0
    running_itc  = 0.0
    running_itm  = 0.0

    pbar = tqdm(dataloader, disable=(RANK != 0), desc=f"train epoch {epoch}")

    for pixel_values, input_ids, attention_mask in pbar:
        pixel_values   = pixel_values.to(DEVICE, non_blocking=True)
        input_ids      = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)

        # unwrap DDP to access helper methods
        core_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        with torch.amp.autocast("cuda"):
            # ITC (contrastive alignment)
            _, _, logits_itc = core_model.get_global_embeddings_and_logits(
                pixel_values, input_ids, attention_mask
            )
            loss_itc = itc_loss_from_logits(logits_itc)

            # ITM (match vs mismatch classification with fusion)
            logits_all, labels_all = build_itm_batch(
                core_model, pixel_values, input_ids, attention_mask
            )
            loss_itm = itm_loss_from_logits(logits_all, labels_all)

            loss = loss_itc + loss_itm

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # cosine LR update
        lr_val = cosine_lr(global_step, WARMUP_STEPS, total_steps, LR)
        for g in optimizer.param_groups:
            g["lr"] = lr_val

        running_loss += loss.item()
        running_itc  += loss_itc.item()
        running_itm  += loss_itm.item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "itc":  f"{loss_itc.item():.4f}",
            "itm":  f"{loss_itm.item():.4f}",
            "lr":   f"{lr_val:.2e}",
        })

        global_step += 1

    n_batches = len(dataloader)
    avg_loss = running_loss / n_batches
    avg_itc  = running_itc  / n_batches
    avg_itm  = running_itm  / n_batches

    if RANK == 0:
        print(f"[epoch {epoch}] train_loss={avg_loss:.4f} itc={avg_itc:.4f} itm={avg_itm:.4f}")

    return global_step, avg_loss


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    sampler = dataloader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(0)

    all_losses = []
    correct_i2t = 0
    correct_t2i = 0
    total_pairs = 0

    pbar = tqdm(dataloader, disable=(RANK != 0), desc="val")

    for pixel_values, input_ids, attention_mask in pbar:
        pixel_values   = pixel_values.to(DEVICE, non_blocking=True)
        input_ids      = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)

        # unwrap DDP
        core_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        with torch.amp.autocast("cuda"):
            # ITC
            _, _, logits_itc = core_model.get_global_embeddings_and_logits(
                pixel_values, input_ids, attention_mask
            )
            loss_itc = itc_loss_from_logits(logits_itc)

            # ITM
            logits_all, labels_all = build_itm_batch(
                core_model, pixel_values, input_ids, attention_mask
            )
            loss_itm = itm_loss_from_logits(logits_all, labels_all)

            loss = loss_itc + loss_itm
            all_losses.append(loss.item())

            # retrieval metrics @1
            B = logits_itc.size(0)

            preds_i2t = logits_itc.argmax(dim=1)  # best text for each image
            correct_i2t += (preds_i2t == torch.arange(B, device=DEVICE)).sum().item()

            preds_t2i = logits_itc.argmax(dim=0)  # best image for each text
            correct_t2i += (preds_t2i == torch.arange(B, device=DEVICE)).sum().item()

            total_pairs += B

    avg_loss = sum(all_losses) / len(all_losses)
    i2t_at1 = correct_i2t / total_pairs
    t2i_at1 = correct_t2i / total_pairs

    if RANK == 0:
        print(f"[val] loss={avg_loss:.4f} i2t@1={i2t_at1:.3f} t2i@1={t2i_at1:.3f}")

    return avg_loss, i2t_at1, t2i_at1


def save_checkpoint(model, optimizer, step, epoch, path):
    # unwrap DDP when saving
    if isinstance(model, nn.parallel.DistributedDataParallel):
        sd = model.module.state_dict()
    else:
        sd = model.state_dict()

    ckpt = {
        "model": sd,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    torch.save(ckpt, path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    torch.cuda.set_device(DEVICE)

    if WORLD_SIZE > 1:
        torch.distributed.init_process_group(backend="nccl")

    # datasets
    train_ds = ImageTextDataset(
        TRAIN_JSON, IMAGE_ROOT, text_tokenizer, train_transform,
        text_key=TEXT_COL, img_key=IMG_COL
    )
    val_ds = ImageTextDataset(
        VAL_JSON, IMAGE_ROOT, text_tokenizer, val_transform,
        text_key=TEXT_COL, img_key=IMG_COL
    )

    # samplers
    if WORLD_SIZE > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
        shuffle_flag  = False
    else:
        train_sampler = None
        val_sampler   = None
        shuffle_flag  = True

    # loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=shuffle_flag,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    # model
    model = ALBEFModel(itc_dim=ITC_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)

    if WORLD_SIZE > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            find_unused_parameters=False
        )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler()

    total_steps = len(train_dl) * EPOCHS
    global_step = 0
    best_val = float("inf")

    for epoch in range(EPOCHS):
        global_step, train_loss = train_one_epoch(
            model, train_dl, optimizer, epoch, global_step, total_steps, scaler
        )

        val_loss, i2t_at1, t2i_at1 = evaluate(model, val_dl)

        if RANK == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"step{global_step}_epoch{epoch}.pt")
            save_checkpoint(model, optimizer, global_step, epoch, ckpt_path)

            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(SAVE_DIR, "best.pt")
                save_checkpoint(model, optimizer, global_step, epoch, best_path)
                print(f"[epoch {epoch}] 🔥 new best, saved to {best_path}")

    if WORLD_SIZE > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()