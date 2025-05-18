import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPModel
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from dataloader import get_loaders
import matplotlib.cm as cm
from collections import defaultdict

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# === Model Definition ===
class ImagePrefixCaptioner(nn.Module):
    def __init__(self, vision_model, language_model, proj_dim):
        super().__init__()
        self.vision_encoder = vision_model.vision_model
        self.language_model = language_model
        self.proj = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim)

    def get_frame_embedding(self, pixel_values):
        with torch.no_grad():
            image_features = self.vision_encoder(pixel_values).last_hidden_state[:, 0, :]
        image_embeds = self.proj(image_features)
        return image_embeds

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            image_features = self.vision_encoder(pixel_values).last_hidden_state[:, 0, :]
        image_embeds = self.proj(image_features)

        inputs_embeds = self.language_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([image_embeds.unsqueeze(1), inputs_embeds], dim=1)

        prefix_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=torch.cat([torch.full((labels.shape[0], 1), -100, device=labels.device), labels], dim=1)
        )
        return outputs

# === Load Models ===
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
language_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

model = ImagePrefixCaptioner(vision_model, language_model, language_model.config.n_embd).to(device)
model.eval()

# === Load Data ===
train_loader, _ = get_loaders(tokenizer)

# === Collect Embeddings and Captions ===
all_embeddings = []
all_captions = []

with torch.no_grad():
    for i,batch in enumerate(tqdm(train_loader, desc="Extracting embeddings")):
        if i >= 5:
            break
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]
        embeds = model.get_frame_embedding(pixel_values).cpu()
        all_embeddings.append(embeds)
        for lbl in labels:
            all_captions.append(tokenizer.decode(lbl, skip_special_tokens=True))

# === Stack All Embeddings ===
all_embeddings_tensor = torch.cat(all_embeddings, dim=0).numpy()

# === Run t-SNE ===
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
reduced = tsne.fit_transform(all_embeddings_tensor)

# === Label Extraction for Coloring ===
def extract_label(caption):
    return caption.strip().split()[0].lower() if caption.strip() else "unknown"

labels = [extract_label(c) for c in all_captions]
unique_labels = sorted(set(labels))
label_to_color = {label: cm.tab20(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

# === Plotting ===
plt.figure(figsize=(12, 8))
label_counts = defaultdict(int)

for i, (x, y) in enumerate(reduced):
    label = labels[i]
    color = label_to_color[label]
    plt.scatter(x, y, color=color, label=label if label_counts[label] == 0 else "", alpha=0.7)
    label_counts[label] += 1
    plt.text(x + 0.5, y, all_captions[i][:25], fontsize=6)

# === Legend ===
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color[l], label=l, markersize=6)
           for l in unique_labels]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Caption Start")

plt.title("t-SNE of Frame Embeddings with Color-Coded Captions")
plt.grid(True)
plt.tight_layout()
plt.show()
