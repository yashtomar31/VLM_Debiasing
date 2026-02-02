import os, shutil
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel

# ---------------- CONFIG ----------------
CSV_PATH = "./test.csv"
CKPTS = {
    "baseline1": "./clip_model_baseline1.pth",
    "baseline2": "./clip_model_baseline2.pth",
    "baseline3": "./clip_model_baseline3.pth",
}
TEXT_BACKBONE = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_IMG = 16
BATCH_TXT = 16
OUT_DIR = "./retrieved_images"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def build_img_id(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3:
        return "_".join(parts[-3:])  # last 2 subfolders + filename
    elif len(parts) >= 2:
        return "_".join(parts[-2:])
    else:
        return os.path.basename(path)

def safe_copy(src_path, dst_dir):
    if not os.path.exists(src_path):
        return
    dst_name = build_img_id(src_path)
    dst_path = os.path.join(dst_dir, dst_name)
    if not os.path.exists(dst_path):
        try:
            shutil.copy(src_path, dst_path)
        except Exception as e:
            print(f"⚠️ Could not copy {src_path}: {e}")

# ---------------- Model ----------------
class CLIPModel(nn.Module):
    def __init__(self, embed_dim=512, text_backbone=TEXT_BACKBONE):
        super().__init__()
        self.image_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, embed_dim)
        self.text_encoder = AutoModel.from_pretrained(text_backbone)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

    def mean_pool(self, out, mask):
        mask = mask.unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    @torch.no_grad()
    def encode_images(self, imgs):
        z = self.image_encoder(imgs)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts, tokenizer, batch_size=BATCH_TXT):
        outs = []
        for i in range(0, len(texts), batch_size):
            tok = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            out = self.text_encoder(**tok)
            pooled = self.mean_pool(out.last_hidden_state, tok.attention_mask)
            z = F.normalize(self.text_proj(pooled), dim=-1)
            outs.append(z.cpu())
        return torch.cat(outs, dim=0).numpy()

def load_clip(path):
    model = CLIPModel().to(DEVICE)
    sd = torch.load(path, map_location=DEVICE)
    if "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# ---------------- Data ----------------
class ImgDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(map(str, paths))
        self.tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)

def encode_images(model, paths):
    ds = ImgDataset(paths)
    dl = DataLoader(ds, batch_size=BATCH_IMG, shuffle=False)
    outs = []
    with torch.no_grad():
        for imgs in dl:
            imgs = imgs.to(DEVICE)
            outs.append(model.encode_images(imgs).cpu())
            torch.cuda.empty_cache()
    return torch.cat(outs).numpy()

# ---------------- Main ----------------
df = pd.read_csv(CSV_PATH)
path_col = "path" if "path" in df.columns else "Image_file_path"
cap_col = "Caption"
info_col = "Image_info_Cleaned" if "Image_info_Cleaned" in df.columns else None

paths = df[path_col].tolist()
captions = df[cap_col].astype(str).tolist()
img_ids = [build_img_id(p) for p in paths]
tokenizer = AutoTokenizer.from_pretrained(TEXT_BACKBONE)

results_i2t, results_t2i = [], []

for bname, ckpt in CKPTS.items():
    print(f"\nEvaluating {bname} ...")
    model = load_clip(ckpt)
    img_emb = encode_images(model, paths)

    # text for retrieval
    if bname == "baseline3" and info_col:
        texts = (df[cap_col].astype(str) + " " + df[info_col].fillna("")).tolist()
    else:
        texts = captions
    txt_emb = model.encode_texts(texts, tokenizer)

    # I->T
    s_i2t = img_emb @ txt_emb.T
    top_i2t = s_i2t.argmax(axis=1)
    for i in range(len(df)):
        j = int(top_i2t[i])
        q_img = paths[i]
        gt_cap = captions[i]
        ret_cap = captions[j]
        q_id = img_ids[i]
        ret_id = img_ids[j]
        results_i2t.append({
            "query_img_id": q_id,
            "GT_caption": gt_cap,
            f"{bname}_Top1_id": ret_id,
            f"{bname}_Top1_caption": ret_cap
        })
        safe_copy(q_img, OUT_DIR)
        safe_copy(paths[j], OUT_DIR)

    # T->I
    s_t2i = txt_emb @ img_emb.T
    top_t2i = s_t2i.argmax(axis=1)
    for i in range(len(df)):
        j = int(top_t2i[i])
        q_cap = captions[i]
        gt_id = img_ids[i]
        ret_id = img_ids[j]
        results_t2i.append({
            "query_caption": q_cap,
            "GT_img_id": gt_id,
            f"{bname}_Top1_img_id": ret_id,
            f"{bname}_Top1_caption": captions[j]
        })
        safe_copy(paths[j], OUT_DIR)
        safe_copy(paths[i], OUT_DIR)

# Merge into DataFrames
i2t_df = pd.DataFrame(results_i2t)
t2i_df = pd.DataFrame(results_t2i)

# Save
i2t_df.to_csv("i2t_top1_table.csv", index=False)
t2i_df.to_csv("t2i_top1_table.csv", index=False)
print("\n✅ Done! Saved:")
print(" - i2t_top1_table.csv")
print(" - t2i_top1_table.csv")
print(f" - Copied all referenced images into: {OUT_DIR}")

# Preview
print("\nI→T Preview:")
display(i2t_df.head(10))
print("\nT→I Preview:")
display(t2i_df.head(10))
