import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    CLIPModel,
    get_scheduler,
)
from transformers.modeling_outputs import BaseModelOutput
from dataloader import get_loaders
import evaluate

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# === CLIP-T5 Model with Projection ===
class CLIPT5Prefix(nn.Module):
    def __init__(self, vision_model, text_model, proj_dim):
        super().__init__()
        self.vision_encoder = vision_model.vision_model
        self.text_model = text_model
        self.proj = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            vision_output = self.vision_encoder(pixel_values)
            vision_embed = vision_output.last_hidden_state[:, 0, :]  # CLS token

        vision_embed = self.proj(vision_embed).unsqueeze(1)

        input_embeds = self.text_model.encoder.embed_tokens(input_ids)
        input_embeds = torch.cat([vision_embed, input_embeds], dim=1)

        vision_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        if labels is not None:
            return self.text_model(
                attention_mask=attention_mask,
                encoder_outputs=BaseModelOutput(last_hidden_state=input_embeds),
                labels=labels,
            )
        else:
            return self.text_model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=input_embeds),
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
            )

# === Load Models ===
tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer.pad_token = tokenizer.eos_token
text_model = T5ForConditionalGeneration.from_pretrained("t5-small")
vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

model = CLIPT5Prefix(
    vision_model, text_model, proj_dim=text_model.config.d_model
).to(device)
model.load_state_dict(torch.load("clip_t5_model.pt", map_location=device))
model.eval()

# === Load Test Data ===
_, test_loader = get_loaders(tokenizer)

# === Evaluation Setup ===
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

predictions = []
references = []

# === Inference Loop ===
for batch in tqdm(test_loader, desc="Evaluating"):
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"]

    generated_ids = model(pixel_values, input_ids, attention_mask)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    predictions.extend(generated_texts)
    references.extend(reference_texts)

# === Evaluate Metrics ===
results = {
    "BLEU": bleu.compute(predictions=predictions, references=[[r] for r in references]),
    "ROUGE": rouge.compute(predictions=predictions, references=references),
    "METEOR": meteor.compute(predictions=predictions, references=references),
}

# === Print Results ===
print("\n📊 Evaluation Results:")
for name, score in results.items():
    if isinstance(score, dict):
        for k, v in score.items():
            if isinstance(v, (float, int)):
                print(f"{name} ({k}): {v:.4f}")
            else:
                print(f"{name} ({k}): {v}")
    else:
        print(f"{name}: {score:.4f}")