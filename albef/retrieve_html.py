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
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
from html import escape

# ===================== CONFIG =====================
CONFIG = {
    "IMAGE_ROOT": "/home/yash/common_pmcoa_data",
    "MODELS": {
        "albef_model1": {
            "ckpt": "./checkpoints_albef_model1_v2/best.pt",
            "test_json": "./test.json",
        },
        "albef_model2": {
            "ckpt": "./checkpoints_albef_model2/best.pt",
            "test_json": "./test.json",
        },
        "albef_model3_v2": {
            "ckpt": "./checkpoints_albef_model3/best.pt",
            "test_json": "./test.json",
        },
    },
    "SUMMARY_CSV": "albef_comparison.csv",
    "DETAILS_CSV": "albef_scores.csv",
    "FIGS_DIR": "figs_albef",
    "CASES_HTML": "albef_model3_wins.html",

    "WEB_IMG_PREFIX": "images",

    "MAX_TEXT_LEN": 64,
    "IMG_SIZE": 224,
    "BATCH_SIZE_IMG": 64,
    "BATCH_SIZE_TXT": 256,
    "SAMPLE_SIZE": 0,
    "RNG_SEED": 1234,

    "HIDDEN_DIM": 768,
    "ITC_DIM": 256,

    "SOFT_PERCENTILE": 0.95,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== HELPERS =====================
def load_json_list(path):
    with open(path, "r") as f:
        return json.load(f)


def load_eval_df(json_path, image_root):
    raw = load_json_list(json_path)
    rows = []
    for row in raw:
        img_rel = row.get("image", "")
        cap = row.get("text_input", "")
        if not img_rel or not cap:
            continue
        img_abs = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
        if os.path.exists(img_abs) and cap.strip() != "":
            rows.append({"Image_file_path": img_abs, "Caption": cap.strip()})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid eval samples for {json_path}")
    return df.reset_index(drop=True)


def df_sample(df, n=0, seed=1234):
    if n and n > 0 and n < len(df):
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


class VisionTransformerTokens(nn.Module):
    def __init__(self, hidden_dim=768):
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

    def forward(self, pixel_values):
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
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.text_model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state
        pooled = tokens.mean(dim=1)
        return pooled


class ALBEFForEval(nn.Module):
    def __init__(self, hidden_dim=768, itc_dim=256):
        super().__init__()
        self.vision_encoder = VisionTransformerTokens(hidden_dim=hidden_dim)
        self.text_encoder = TextEncoder("distilbert-base-uncased")

        self.vision_proj = nn.Linear(hidden_dim, itc_dim)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, itc_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_images(self, pixel_values):
        cls_vec = self.vision_encoder(pixel_values)
        img_emb = self.vision_proj(cls_vec)
        img_emb = F.normalize(img_emb, dim=-1)
        return img_emb

    def encode_texts(self, input_ids, attention_mask):
        txt_vec = self.text_encoder(input_ids, attention_mask)
        txt_emb = self.text_proj(txt_vec)
        txt_emb = F.normalize(txt_emb, dim=-1)
        return txt_emb


def get_image_transform(img_size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_text_tokenizer():
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


@torch.no_grad()
def embed_images_batched(model, image_paths, transform, batch_size):
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
def embed_texts_batched(model, captions, tokenizer, max_len, batch_size):
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


def hit_at_k_binary_vector(sim_mat, k):
    N = sim_mat.size(0)
    k = min(k, N)
    topk_idx = torch.topk(sim_mat, k=k, dim=1).indices.cpu().numpy()
    return np.array([1 if i in topk_idx[i] else 0 for i in range(N)], dtype=np.int32)


def hit_at_k(sim_mat, k):
    return float(hit_at_k_binary_vector(sim_mat, k).mean())


def mean_topk_similarity(sim_mat, k):
    N = sim_mat.size(0)
    if N == 0:
        return 0.0
    k = min(k, N)
    topk_vals = torch.topk(sim_mat, k=k, dim=1).values
    per_query_mean = topk_vals.mean(1).clamp(0, 1)
    return per_query_mean.mean().item()


def per_sample_mean_topk(sim_mat, k):
    k = min(k, sim_mat.size(0))
    vals = torch.topk(sim_mat, k=k, dim=1).values.mean(1).clamp(0, 1)
    return vals.detach().cpu().numpy()


def soft_recall_generic(scores_query_to_cands, ref_emb, k_list=(1, 5, 10), percentile=0.95):
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


@torch.no_grad()
def evaluate_checkpoint_on_its_test(
    model_name,
    ckpt_path,
    test_json,
    image_root,
    img_transform,
    tokenizer,
    hidden_dim,
    itc_dim,
    max_len,
    img_bsz,
    txt_bsz,
    sample_size,
    rng_seed,
    soft_percentile,
    topk_for_html=3,
):
    df = load_eval_df(test_json, image_root)
    if sample_size and sample_size > 0:
        df = df_sample(df, sample_size, rng_seed)
        print(f"[{model_name}] using SAMPLE_SIZE={len(df)}")
    else:
        print(f"[{model_name}] total eval samples={len(df)}")

    captions = df["Caption"].tolist()
    image_paths = df["Image_file_path"].tolist()

    core = ALBEFForEval(hidden_dim=hidden_dim, itc_dim=itc_dim).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = core.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[{model_name}] missing keys:", missing)
    if unexpected:
        print(f"[{model_name}] unexpected keys:", unexpected)

    txt_emb = embed_texts_batched(core, captions, tokenizer, max_len, txt_bsz)
    img_emb = embed_images_batched(core, image_paths, img_transform, img_bsz)

    t2i = txt_emb @ img_emb.T
    i2t = img_emb @ txt_emb.T

    # For HTML: Top-K indices + cosine values
    K_html = min(topk_for_html, t2i.size(0))
    topk = torch.topk(t2i, k=K_html, dim=1)
    t2i_topk_idx = topk.indices.cpu().numpy()
    t2i_topk_vals = topk.values.cpu().numpy()

    row = {"Model": model_name, "NumSamples": t2i.size(0)}
    details_rows = []

    per_sample = {
        "T2I_hit@1": hit_at_k_binary_vector(t2i, 1),
        "T2I_hit@5": hit_at_k_binary_vector(t2i, 5),
        "T2I_hit@10": hit_at_k_binary_vector(t2i, 10),
        "I2T_hit@1": hit_at_k_binary_vector(i2t, 1),
        "I2T_hit@5": hit_at_k_binary_vector(i2t, 5),
        "I2T_hit@10": hit_at_k_binary_vector(i2t, 10),
        "T2I_mSim@1_dist": per_sample_mean_topk(t2i, 1),
        "T2I_mSim@5_dist": per_sample_mean_topk(t2i, 5),
        "T2I_mSim@10_dist": per_sample_mean_topk(t2i, 10),
        "I2T_mSim@1_dist": per_sample_mean_topk(i2t, 1),
        "I2T_mSim@5_dist": per_sample_mean_topk(i2t, 5),
        "I2T_mSim@10_dist": per_sample_mean_topk(i2t, 10),
    }

    for k in (1, 5, 10):
        ke = min(k, t2i.size(0))
        row[f"T2I_hit@{k}"] = round(hit_at_k(t2i, ke), 4)
        row[f"I2T_hit@{k}"] = round(hit_at_k(i2t, ke), 4)
        row[f"T2I_mSim@{k}"] = round(mean_topk_similarity(t2i, ke), 4)
        row[f"I2T_mSim@{k}"] = round(mean_topk_similarity(i2t, ke), 4)

        for v in per_sample_mean_topk(t2i, ke):
            details_rows.append({"Model": model_name, "Direction": "T2I", "K": k, "Score": float(v)})
        for v in per_sample_mean_topk(i2t, ke):
            details_rows.append({"Model": model_name, "Direction": "I2T", "K": k, "Score": float(v)})

    soft_rec_i2t = soft_recall_generic(i2t, txt_emb, (1, 5, 10), soft_percentile)
    for k, val in soft_rec_i2t.items():
        row[f"I2T_softRecall@{k}"] = round(val, 4)

    soft_rec_t2i = soft_recall_generic(t2i, img_emb, (1, 5, 10), soft_percentile)
    for k, val in soft_rec_t2i.items():
        row[f"T2I_softRecall@{k}"] = round(val, 4)

    def soft_flags(scores_query_to_cands, ref_emb, K, percentile):
        N = scores_query_to_cands.size(0)
        K_eff = min(K, N)
        topk_idx = torch.topk(scores_query_to_cands, k=K_eff, dim=1).indices.cpu().numpy()
        sim_within_ref = ref_emb @ ref_emb.T
        flags = []
        for i in range(N):
            sims_row = sim_within_ref[i].detach().cpu().numpy()
            thr = np.quantile(sims_row, percentile)
            retrieved_ids = topk_idx[i]
            retrieved_sims = sims_row[retrieved_ids]
            flags.append(1 if np.any(retrieved_sims >= thr) else 0)
        return np.array(flags, dtype=np.int32)

    for k in (1, 5, 10):
        per_sample[f"T2I_softRecall@{k}"] = soft_flags(t2i, img_emb, k, soft_percentile)
        per_sample[f"I2T_softRecall@{k}"] = soft_flags(i2t, txt_emb, k, soft_percentile)

    extra = {
        "captions": captions,
        "image_paths": image_paths,
        "T2I_topk_idx": t2i_topk_idx,
        "T2I_topk_vals": t2i_topk_vals,
    }

    return row, details_rows, per_sample, extra


# ===================== STATS =====================
def mcnemar_p(baseline_vec, other_vec):
    b1o1 = np.sum((baseline_vec == 1) & (other_vec == 1))
    b1o0 = np.sum((baseline_vec == 1) & (other_vec == 0))
    b0o1 = np.sum((baseline_vec == 0) & (other_vec == 1))
    b0o0 = np.sum((baseline_vec == 0) & (other_vec == 0))

    table = [[b1o1, b1o0],
             [b0o1, b0o0]]
    res = mcnemar(table, exact=True)
    return res.pvalue


def add_soft_pvals_and_stars(summary_df, per_model_samples, baseline_name="albef_model2"):
    base = per_model_samples[baseline_name]
    soft_metrics = [
        "T2I_softRecall@1", "T2I_softRecall@5", "T2I_softRecall@10",
        "I2T_softRecall@1", "I2T_softRecall@5", "I2T_softRecall@10",
    ]

    def stars_for_p(p):
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    pvals = {m: {} for m in soft_metrics}
    for metric in soft_metrics:
        bvec = base[metric]
        for model_name, sample_dict in per_model_samples.items():
            if model_name == baseline_name:
                pvals[metric][model_name] = np.nan
            else:
                pvals[metric][model_name] = mcnemar_p(bvec, sample_dict[metric])

    for metric in soft_metrics:
        p_col = metric + "_p_vs_" + baseline_name
        summary_df[p_col] = summary_df["Model"].map(lambda m: pvals[metric][m])

        def with_stars(row):
            val = row[metric]
            p = row[p_col]
            return f"{val}{stars_for_p(p)}"

        starred_col = metric + "_starred"
        summary_df[starred_col] = summary_df.apply(with_stars, axis=1)

    return summary_df


def paired_t_p(baseline_scores, other_scores):
    stat, p = ttest_rel(baseline_scores, other_scores)
    return p


# ===================== HTML COMPARISON =====================
def make_model3_comparison_html(
    per_model_samples,
    per_model_extra,
    model1="albef_model1",
    model2="albef_model2",
    model3="albef_model3_v2",
    out_html_path="albef_model3_wins.html",
):
    required = [model1, model2, model3]
    for m in required:
        if m not in per_model_samples or m not in per_model_extra:
            print(f"[HTML] Missing data for {m}, skipping HTML.")
            return

    t2i_m1 = per_model_samples[model1]["T2I_hit@1"]
    t2i_m2 = per_model_samples[model2]["T2I_hit@1"]
    t2i_m3 = per_model_samples[model3]["T2I_hit@1"]

    N = len(t2i_m3)
    if not (len(t2i_m1) == len(t2i_m2) == N):
        print("[HTML] Length mismatch in per-sample vectors, skipping HTML.")
        return

    extra_m1 = per_model_extra[model1]
    extra_m2 = per_model_extra[model2]
    extra_m3 = per_model_extra[model3]

    caps_m1 = extra_m1["captions"]
    caps_m2 = extra_m2["captions"]
    caps_m3 = extra_m3["captions"]

    paths_m1 = extra_m1["image_paths"]
    paths_m2 = extra_m2["image_paths"]
    paths_m3 = extra_m3["image_paths"]

    if not (len(caps_m1) == len(caps_m2) == len(caps_m3) == N):
        print("[HTML] Caption lengths differ between models, skipping HTML.")
        return
    if not (len(paths_m1) == len(paths_m2) == len(paths_m3) == N):
        print("[HTML] Image list lengths differ between models, skipping HTML.")
        return

    image_paths = paths_m3

    mask = (t2i_m3 == 1) & ((t2i_m1 == 0) | (t2i_m2 == 0))
    indices = np.where(mask)[0]
    if len(indices) == 0:
        print("[HTML] No cases where model3 is correct and at least one of model1/model2 is wrong.")
        return

    topk_m1 = extra_m1["T2I_topk_idx"]
    topk_m2 = extra_m2["T2I_topk_idx"]
    topk_m3 = extra_m3["T2I_topk_idx"]

    vals_m1 = extra_m1["T2I_topk_vals"]
    vals_m2 = extra_m2["T2I_topk_vals"]
    vals_m3 = extra_m3["T2I_topk_vals"]

    def to_web_path(abs_path):
        rel = os.path.relpath(abs_path, start=CONFIG["IMAGE_ROOT"])
        return f"{CONFIG['WEB_IMG_PREFIX']}/{rel}"

    html_lines = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html>")
    html_lines.append("<head>")
    html_lines.append("<meta charset='utf-8'/>")
    html_lines.append(f"<title>{escape(model3)} comparison vs {escape(model1)} &amp; {escape(model2)}</title>")
    html_lines.append("""
<style>
body { font-family: Arial, sans-serif; }
.table-wrapper { overflow-x: auto; }
.case-table { border-collapse: collapse; width: 100%; min-width: 1400px; }
.case-table th, .case-table td {
  border: 1px solid #ccc; padding: 8px; vertical-align: top; text-align: center;
}
.model-header { background-color: #f0f0f0; font-weight: bold; }
.img-cell img {
  max-width: 200px; max-height: 200px; object-fit: contain; display: block; margin: 0 auto 4px auto;
}
.caption-text { font-size: 12px; }
.sim-text { font-size: 12px; color: #888; margin-top: 2px; }
.query-info { font-size: 12px; color: #555; margin-top: 4px; }
.top1.correct { border: 2px solid #2e7d32; }
.top1.wrong { border: 2px solid #c62828; }
</style>
""")
    html_lines.append("</head>")
    html_lines.append("<body>")
    html_lines.append(
        f"<h1>Cases where <code>{escape(model3)}</code> is correct (T2I@1)"
        f" and at least one of <code>{escape(model1)}</code>, <code>{escape(model2)}</code> is wrong</h1>"
    )
    html_lines.append(
        "<p>Images are the same across models, but captions differ:"
        "<br/>M1 &amp; M2 use captions from <code>test.json</code>,"
        "<br/>M3 uses captions from <code>test_model3.json</code>.</p>"
    )
    html_lines.append(f"<p>Total cases: {len(indices)}</p>")

    html_lines.append("<div class='table-wrapper'>")
    html_lines.append("<table class='case-table'>")

    html_lines.append("<tr>")
    html_lines.append("<th rowspan='2'>Query<br/>(GT image + captions)</th>")
    html_lines.append(f"<th class='model-header' colspan='3'>{escape(model1)}</th>")
    html_lines.append(f"<th class='model-header' colspan='3'>{escape(model2)}</th>")
    html_lines.append(f"<th class='model-header' colspan='3'>{escape(model3)}</th>")
    html_lines.append("</tr>")

    html_lines.append("<tr>")
    html_lines.append("<th>#1</th><th>#2</th><th>#3</th>")
    html_lines.append("<th>#1</th><th>#2</th><th>#3</th>")
    html_lines.append("<th>#1</th><th>#2</th><th>#3</th>")
    html_lines.append("</tr>")

    def add_model_cells(idx, topk, topk_vals, t2i_vec, img_paths, caps):
        ranks = topk[idx]
        sims = topk_vals[idx]
        for r, img_idx in enumerate(ranks):
            cand_abs = img_paths[img_idx]
            cand_web = to_web_path(cand_abs)
            cand_caption = caps[img_idx]
            sim_score = float(sims[r])

            classes = "img-cell"
            if r == 0:
                classes += " top1 correct" if t2i_vec[idx] == 1 else " top1 wrong"

            html_lines.append(f"<td><div class='{classes}'>")
            html_lines.append(f"<img src='{cand_web}' alt='Rank {r+1}'/>")
            html_lines.append(f"<div class='caption-text'>{escape(cand_caption)}</div>")
            html_lines.append(f"<div class='sim-text'>cos: {sim_score:.4f}</div>")
            html_lines.append("</div></td>")

    for idx in indices:
        gt_abs = image_paths[idx]
        gt_web = to_web_path(gt_abs)

        html_lines.append("<tr>")

        html_lines.append("<td>")
        html_lines.append("<div class='img-cell'>")
        html_lines.append(f"<img src='{gt_web}' alt='Ground truth'/>")
        html_lines.append("</div>")
        html_lines.append("<div class='caption-text'><strong>Caption (M1/M2):</strong> "
                          f"{escape(caps_m1[idx])}</div>")
        html_lines.append("<div class='caption-text'><strong>Caption (M3):</strong> "
                          f"{escape(caps_m3[idx])}</div>")
        html_lines.append(
            "<div class='query-info'>"
            f"M1: {'correct' if t2i_m1[idx] == 1 else 'wrong'}, "
            f"M2: {'correct' if t2i_m2[idx] == 1 else 'wrong'}, "
            f"M3: {'correct' if t2i_m3[idx] == 1 else 'wrong'}"
            "</div>"
        )
        html_lines.append("</td>")

        add_model_cells(idx, topk_m1, vals_m1, t2i_m1, paths_m1, caps_m1)
        add_model_cells(idx, topk_m2, vals_m2, t2i_m2, paths_m2, caps_m2)
        add_model_cells(idx, topk_m3, vals_m3, t2i_m3, paths_m3, caps_m3)

        html_lines.append("</tr>")

    html_lines.append("</table>")
    html_lines.append("</div>")
    html_lines.append("</body></html>")

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print(f"[HTML] Saved comparison table to: {out_html_path}")


# ===================== PLOTTING =====================
def _grouped_bar(ax, means_df, title):
    ks = [1, 5, 10]
    models = list(means_df["Model"].unique())
    x = np.arange(len(ks), dtype=float)

    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        vals = [
            means_df[(means_df["Model"] == m) & (means_df["K"] == k)]["ScoreMean"].mean()
            for k in ks
        ]
        ax.bar(x + (i - (len(models) - 1) / 2.0) * width, vals, width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(title="Model")


def _boxplot(ax, dist_df, title):
    ks = [1, 5, 10]
    models = list(dist_df["Model"].unique())

    positions, data, tick_positions = [], [], []
    pos = 1.0
    gap = 1.0

    for k in ks:
        for m in models:
            vals = dist_df[(dist_df["Model"] == m) & (dist_df["K"] == k)]["Score"].values
            data.append(vals)
            positions.append(pos)
            pos += 1.0
        tick_positions.append(np.mean(positions[-len(models):]))
        pos += gap

    ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=False,
        manage_ticks=False,
        showfliers=True,
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)

    ax.legend([plt.Line2D([0], [0]) for _ in models], models, title="Model")


def make_plots(scores_csv: str, summary_csv: str, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_csv)
    needed = {"Model", "Direction", "K", "Score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {scores_csv}: {missing}")

    for direction in ["T2I", "I2T"]:
        sub = df[df["Direction"] == direction].copy()
        means = (
            sub.groupby(["K", "Model"])["Score"]
               .mean()
               .reset_index()
               .rename(columns={"Score": "ScoreMean"})
        )

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        _grouped_bar(ax, means, f"{direction} Mean Recall@K (Mean Similarity)")
        fig.tight_layout()
        fig.savefig(out / f"{direction}_mean_recall_at_k.png")
        plt.close(fig)

    for direction in ["T2I", "I2T"]:
        sub = df[df["Direction"] == direction].copy()
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        _boxplot(ax, sub[["K", "Model", "Score"]], f"{direction} Recall@K Distribution (Mean Similarity)")
        fig.tight_layout()
        fig.savefig(out / f"{direction}_recall_at_k_boxplot.png")
        plt.close(fig)

    summ = pd.read_csv(summary_csv)
    cols = [
        "Model",
        "T2I_hit@1", "T2I_hit@5", "T2I_hit@10",
        "I2T_hit@1", "I2T_hit@5", "I2T_hit@10",
        "T2I_mSim@1", "T2I_mSim@5", "T2I_mSim@10",
        "I2T_mSim@1", "I2T_mSim@5", "I2T_mSim@10",
        "T2I_softRecall@1", "T2I_softRecall@5", "T2I_softRecall@10",
        "I2T_softRecall@1", "I2T_softRecall@5", "I2T_softRecall@10",
    ]
    have = [c for c in cols if c in summ.columns]
    if have:
        print("\nSummary (means):")
        print(summ[have].to_string(index=False))


# ===================== MAIN =====================
def main():
    torch.set_grad_enabled(False)

    img_transform = get_image_transform(CONFIG["IMG_SIZE"])
    tokenizer = get_text_tokenizer()

    rows_all = []
    details_all = []
    per_model_samples = {}
    per_model_extra = {}

    for model_name, spec in CONFIG["MODELS"].items():
        ckpt_path = spec["ckpt"]
        test_json = spec["test_json"]
        image_root = CONFIG["IMAGE_ROOT"]

        print(f"\n=== Evaluating {model_name} on {test_json} ===")

        row, details, per_sample, extra = evaluate_checkpoint_on_its_test(
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
            topk_for_html=3,
        )

        rows_all.append(row)
        details_all.extend(details)
        per_model_samples[model_name] = per_sample
        per_model_extra[model_name] = extra

    baseline_name = "albef_model2"

    summary_cols = [
        "Model", "NumSamples",
        "T2I_hit@1", "T2I_hit@5", "T2I_hit@10",
        "I2T_hit@1", "I2T_hit@5", "I2T_hit@10",
        "T2I_softRecall@1", "T2I_softRecall@5", "T2I_softRecall@10",
        "I2T_softRecall@1", "I2T_softRecall@5", "I2T_softRecall@10",
    ]

    summary_df = (
        pd.DataFrame(rows_all)[summary_cols]
          .sort_values("T2I_softRecall@10", ascending=False)
          .reset_index(drop=True)
    )

    summary_df = add_soft_pvals_and_stars(summary_df, per_model_samples, baseline_name=baseline_name)

    pretty_cols = [
        "Model",
        "T2I_softRecall@1_starred",  f"T2I_softRecall@1_p_vs_{baseline_name}",
        "T2I_softRecall@5_starred",  f"T2I_softRecall@5_p_vs_{baseline_name}",
        "T2I_softRecall@10_starred", f"T2I_softRecall@10_p_vs_{baseline_name}",
        "I2T_softRecall@1_starred",  f"I2T_softRecall@1_p_vs_{baseline_name}",
        "I2T_softRecall@5_starred",  f"I2T_softRecall@5_p_vs_{baseline_name}",
        "I2T_softRecall@10_starred", f"I2T_softRecall@10_p_vs_{baseline_name}",
    ]

    print("\n=== SOFT RECALL Comparison (semantic match) ===")
    print(summary_df[pretty_cols].to_string(index=False))

    Path(CONFIG["FIGS_DIR"]).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(CONFIG["SUMMARY_CSV"], index=False)
    pd.DataFrame(details_all).to_csv(CONFIG["DETAILS_CSV"], index=False)

    print(f"\nSaved summary: {CONFIG['SUMMARY_CSV']}")
    print(f"Saved per-sample scores: {CONFIG['DETAILS_CSV']}")

    make_plots(CONFIG["DETAILS_CSV"], CONFIG["SUMMARY_CSV"], CONFIG["FIGS_DIR"])
    print(f"Figures saved to: {CONFIG['FIGS_DIR']}")

    make_model3_comparison_html(
        per_model_samples=per_model_samples,
        per_model_extra=per_model_extra,
        model1="albef_model1",
        model2="albef_model2",
        model3="albef_model3_v2",
        out_html_path=CONFIG["CASES_HTML"],
    )


if __name__ == "__main__":
    main()