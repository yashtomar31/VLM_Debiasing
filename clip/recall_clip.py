import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
BATCH_IMG = 16      # smaller = less VRAM
BATCH_TXT = 16
K_LIST = [1,5,10]
PERCENTILE = 0.95
0

# ---------------- MODEL ----------------
class CLIPModel(torch.nn.Module):
    def __init__(self, embed_dim=512, text_backbone=TEXT_BACKBONE):
        super().__init__()
        self.image_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.image_encoder.fc = torch.nn.Linear(self.image_encoder.fc.in_features, embed_dim)
        self.text_encoder = AutoModel.from_pretrained(text_backbone)
        self.text_proj = torch.nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

    def mean_pool(self, out, mask):
        mask = mask.unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    @torch.no_grad()
    def encode_images(self, imgs):
        return F.normalize(self.image_encoder(imgs), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts, tokenizer):
        outs = []
        for i in range(0, len(texts), BATCH_TXT):
            tok = tokenizer(texts[i:i+BATCH_TXT], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            out = self.text_encoder(**tok)
            pooled = self.mean_pool(out.last_hidden_state, tok.attention_mask)
            z = F.normalize(self.text_proj(pooled), dim=-1)
            outs.append(z.cpu())
        return torch.cat(outs).numpy()

def load_clip(path):
    model = CLIPModel().to(DEVICE)
    sd = torch.load(path, map_location=DEVICE)
    if "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k,v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# ---------------- DATA ----------------
class ImgDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.tf(Image.open(self.paths[i]).convert("RGB"))

def encode_images(model, paths):
    ds = ImgDataset(paths)
    dl = DataLoader(ds, batch_size=BATCH_IMG, shuffle=False)
    out = []
    with torch.no_grad():
        for imgs in dl:
            imgs = imgs.to(DEVICE)
            out.append(model.encode_images(imgs).cpu())
            torch.cuda.empty_cache()  # free memory
    return torch.cat(out).numpy()

# ---------------- SOFT RECALL@K ----------------
def soft_recall_at_k(scores_I2T, cap_emb, k_list, percentile=0.7):
    cap_sim = cap_emb @ cap_emb.T
    recalls = {}
    for K in k_list:
        succ = []
        topk = np.argpartition(-scores_I2T, K-1, axis=1)[:, :K]
        for i in range(len(scores_I2T)):
            thr = np.quantile(cap_sim[i], percentile)
            rel = cap_sim[i, topk[i]]
            succ.append(int(np.any(rel >= thr)))
        recalls[K] = np.mean(succ)
    return recalls

# ---------------- MAIN ----------------
df = pd.read_csv(CSV_PATH)
paths = df["path"].tolist()
captions = df["Caption"].astype(str).tolist()

# Caption embeddings for relevance
tok = AutoTokenizer.from_pretrained(TEXT_BACKBONE)
tmp_model = CLIPModel().to(DEVICE)
tmp_model.eval()
cap_emb = tmp_model.encode_texts(captions, tok)
del tmp_model
torch.cuda.empty_cache()

# Evaluate all baselines
results = {}
for name, ckpt in CKPTS.items():
    print(f"\n{name} ...")
    model = load_clip(ckpt)
    img_emb = encode_images(model, paths)
    if name == "baseline3" and "Image_info_Cleaned" in df.columns:
        texts = (df["Caption"] + " " + df["Image_info_Cleaned"].fillna("")).tolist()
    else:
        texts = captions
    txt_emb = model.encode_texts(texts, tok)
    torch.cuda.empty_cache()
    scores_I2T = img_emb @ txt_emb.T
    recalls = soft_recall_at_k(scores_I2T, cap_emb, K_LIST, percentile=PERCENTILE)
    results[name] = recalls

print("\n======= Soft Recall@K =======")
for K in K_LIST:
    print(f"\nRecall@{K}:")
    for name in results:
        print(f"  {name:<10s}: {results[name][K]:.4f}")

best = {K: max(results, key=lambda n: results[n][K]) for K in K_LIST}
print("\nBest baseline per K:")
for K in K_LIST:
    print(f"  @K={K}: {best[K]} ({results[best[K]][K]:.4f})")
