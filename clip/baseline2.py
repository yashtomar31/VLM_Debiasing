import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ---------- Config ----------
CSV_PATH = "../csv/med_figs_data.csv"  # Your cleaned CSV path
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 512
SAVE_PLOT_PATH = "../loss/loss_baseline2_curve.png"

# ---------- Dataset ----------
class FigureCaptionDataset(Dataset):
    def __init__(self, csv_path, tokenizer, transform):
        df_med = pd.read_csv(CSV_PATH)
        # Remove invalid image paths and captions
        df_med = df_med[~df_med['Image_file_path'].isin(['No images.', '', None])]
        df_med = df_med.dropna(subset=['Image_file_path', 'Caption']).reset_index(drop=True)
        train_df, test_df = train_test_split(df_med, test_size=0.1, random_state=42)
        df = train_df
        print(df.head())
        print(df.shape)
        self.data=df
        self.data = df
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['Image_file_path']
        caption = self.data.iloc[idx]['Caption']

        image = self.transform(Image.open(image_path).convert('RGB'))
        tokens = self.tokenizer(caption, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=64)

        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        return image, input_ids, attention_mask

# ---------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ---------- Data ----------
dataset = FigureCaptionDataset(CSV_PATH, tokenizer, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------- CLIP Model ----------
class CLIPModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, embed_dim)

        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

    def forward(self, image, input_ids, attention_mask):
        img_features = self.image_encoder(image)

        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_out.last_hidden_state.mean(dim=1)
        text_features = self.text_proj(text_features)

        img_features = F.normalize(img_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return img_features, text_features

# ---------- CLIP Loss ----------
def clip_loss(image_embeds, text_embeds, temperature=0.07):
    logits_per_image = image_embeds @ text_embeds.T
    logits_per_text = text_embeds @ image_embeds.T
    labels = torch.arange(len(image_embeds)).to(DEVICE)

    loss_i = F.cross_entropy(logits_per_image / temperature, labels)
    loss_t = F.cross_entropy(logits_per_text / temperature, labels)
    return (loss_i + loss_t) / 2

# ---------- Training ----------
model = CLIPModel(EMBED_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
losses = []

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, input_ids, attention_mask in loop:
        images = images.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        image_embeds, text_embeds = model(images, input_ids, attention_mask)
        loss = clip_loss(image_embeds, text_embeds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")


# ---------- Save Model ----------
MODEL_SAVE_PATH = "../model/clip_model_baseline2.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Saved model to {MODEL_SAVE_PATH}")

# ---------- Save Loss Plot ----------
plt.figure(figsize=(8, 5))
plt.plot(losses, marker='o')
plt.title("Training Loss Curve (With Attention)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(SAVE_PLOT_PATH)
print(f"✅ Saved loss plot to {SAVE_PLOT_PATH}")
