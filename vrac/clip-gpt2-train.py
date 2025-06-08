# train.py
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel
from transformers import get_scheduler
from data.dataloader import get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


class ImagePrefixCaptioner(nn.Module):
    def __init__(self, vision_model, language_model, proj_dim):
        super().__init__()
        self.vision_encoder = vision_model.vision_model
        self.language_model = language_model
        self.proj = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            image_features = self.vision_encoder(pixel_values).last_hidden_state[:, 0, :]  # CLS token
        image_embeds = self.proj(image_features)  # shape: (B, hidden)

        # Embed text tokens
        inputs_embeds = self.language_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([image_embeds.unsqueeze(1), inputs_embeds], dim=1)

        # Adjust attention mask
        prefix_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=torch.cat([torch.full((labels.shape[0], 1), -100, device=labels.device), labels], dim=1)
        )

        return outputs


def train():
    # === Load models ===
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    language_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    model = ImagePrefixCaptioner(
        vision_model=vision_model,
        language_model=language_model,
        proj_dim=language_model.config.n_embd
    ).to(device)

    # === Load data ===
    train_loader, val_loader = get_loaders(tokenizer)

    # === Optimizer & Scheduler ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * 3
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # === Training loop ===
    model.train()
    for epoch in range(3):
        print("Epcoh :")
        total_loss = 0
        for batch in tqdm(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    # === Save model ===
    torch.save(model.state_dict(), "captioning_model.pt")


if __name__ == "__main__":
    train()
