import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from transformers import (
    CLIPModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers.modeling_outputs import BaseModelOutput
from data.dataloader import get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# === CLIP-GPT2 Model ===
class CLIPGPT2(nn.Module):
    def __init__(self, vision_model, language_model, proj_dim):
        super().__init__()
        self.vision_encoder = vision_model.vision_model
        self.language_model = language_model
        self.proj = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim)

    def get_frame_embedding(self, pixel_values):
        with torch.no_grad():
            features = self.vision_encoder(pixel_values).last_hidden_state[:, 0, :]
        return self.proj(features)

    def generate(self, pixel_values, tokenizer, max_length=32, num_beams=4):
        image_embed = self.get_frame_embedding(pixel_values)
        input_ids = torch.full((pixel_values.size(0), 1), tokenizer.bos_token_id, device=device)
        input_embed = self.language_model.transformer.wte(input_ids)
        input_embed = torch.cat([image_embed.unsqueeze(1), input_embed], dim=1)
        attention_mask = torch.ones(input_embed.shape[:2], device=device)

        return self.language_model.generate(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
        )

# === CLIP-T5 Model ===
class CLIPT5(nn.Module):
    def __init__(self, vision_model, text_model, proj_dim):
        super().__init__()
        self.vision_encoder = vision_model.vision_model
        self.text_model = text_model
        self.proj = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim)

    def generate(self, pixel_values, input_ids, attention_mask, tokenizer, max_length=64, num_beams=4):
        with torch.no_grad():
            vision_embed = self.vision_encoder(pixel_values).last_hidden_state[:, 0, :]
        vision_embed = self.proj(vision_embed).unsqueeze(1)

        input_embed = self.text_model.encoder.embed_tokens(input_ids)
        input_embed = torch.cat([vision_embed, input_embed], dim=1)

        vision_mask = torch.ones((attention_mask.size(0), 1), device=device)
        attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        return self.text_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=input_embed),
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
        )

# === Load Tokenizers ===
gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_tokenizer.pad_token = t5_tokenizer.eos_token

# === Load Models ===
clip_model_1 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model_2 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

gpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

gpt2_captioner = CLIPGPT2(clip_model_1, gpt2_model, gpt2_model.config.n_embd).to(device)
gpt2_captioner.load_state_dict(torch.load("captioning_model.pt", map_location=device))
gpt2_captioner.eval()

t5_captioner = CLIPT5(clip_model_2, t5_model, t5_model.config.d_model).to(device)
t5_captioner.load_state_dict(torch.load("clip_t5_model.pt", map_location=device))
t5_captioner.eval()

# === Load Data (separately for each tokenizer) ===
_, gpt2_test_loader = get_loaders(gpt2_tokenizer)
_, t5_test_loader = get_loaders(t5_tokenizer)

gpt2_batch = next(iter(gpt2_test_loader))
t5_batch = next(iter(t5_test_loader))

# === Get Inputs ===
pixel_values = gpt2_batch["pixel_values"].to(device)
gpt2_labels = gpt2_batch["labels"]

t5_input_ids = t5_batch["input_ids"].to(device)
t5_attention_mask = t5_batch["attention_mask"].to(device)
t5_pixel_values = t5_batch["pixel_values"].to(device)
t5_labels = t5_batch["labels"]

# === Generate Captions ===
gpt2_outputs = gpt2_captioner.generate(pixel_values, gpt2_tokenizer)
t5_outputs = t5_captioner.generate(t5_pixel_values, t5_input_ids, t5_attention_mask, t5_tokenizer)

gpt2_preds = gpt2_tokenizer.batch_decode(gpt2_outputs, skip_special_tokens=True)
t5_preds = t5_tokenizer.batch_decode(t5_outputs, skip_special_tokens=True)

gpt2_refs = gpt2_tokenizer.batch_decode(gpt2_labels, skip_special_tokens=True)
t5_refs = t5_tokenizer.batch_decode(t5_labels, skip_special_tokens=True)

# === Visualization ===
for i in range(min(6, len(pixel_values))):
    img = to_pil_image(pixel_values[i])  # Fixed: Use pixel_values instead of gpt2_batch
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Ref:  {gpt2_refs[i]}\nGPT2: {gpt2_preds[i]}\nT5:   {t5_preds[i]}", fontsize=8)
    plt.tight_layout()
    plt.show()

# === Visualization ===
for i in range(min(6, len(pixel_values))):
    img = to_pil_image(gpt2_batch["pixel_values"][i])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Ref:  {gpt2_refs[i]}\nGPT2: {gpt2_preds[i]}\nT5:   {t5_preds[i]}", fontsize=8)
    plt.tight_layout()
    plt.show()
