import torch
import torch.nn as nn
from transformers import CLIPModel, T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from data.dataloader import get_loaders
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class CLIPT5Prefix(nn.Module):
    def __init__(self, vision_model, text_model, proj_dim):
        super().__init__()
        self.vision_encoder = vision_model.vision_model
        self.text_model = text_model
        self.proj = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        with torch.no_grad():
            vision_output = self.vision_encoder(pixel_values)
            vision_embed = vision_output.last_hidden_state[:, 0, :]  # CLS token

        vision_embed = self.proj(vision_embed).unsqueeze(1)  # (B, 1, D)

        # Embed text input
        input_embeds = self.text_model.encoder.embed_tokens(input_ids)
        input_embeds = torch.cat([vision_embed, input_embeds], dim=1)

        # Adjust attention
        vision_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        outputs = self.text_model(
            attention_mask=attention_mask,
            encoder_outputs=(input_embeds,),
            labels=labels
        )

        return outputs

# === Load models ===
tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer.pad_token = tokenizer.eos_token
text_model = T5ForConditionalGeneration.from_pretrained("t5-small")
vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

model = CLIPT5Prefix(vision_model, text_model, proj_dim=text_model.config.d_model).to(device)

# === Load Data ===
train_loader, _ = get_loaders(tokenizer)

# === Training Setup ===
learning_rate = 1e-4
num_epochs = 20
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(train_loader) * num_epochs

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=200,
    num_training_steps=num_training_steps
)

# === Training Loop ===
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values, input_ids, attention_mask, labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional: clip exploding gradients
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# === Save model ===
torch.save(model.state_dict(), "clip_t5_model.pt")