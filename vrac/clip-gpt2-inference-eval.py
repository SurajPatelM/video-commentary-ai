import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPModel
from data.dataloader import get_loaders
from transformers import logging
import evaluate

logging.set_verbosity_error()  # Suppress warnings

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# === Model Class ===
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

    def generate_caption(self, pixel_values, max_length=32, num_beams=4):
        image_embeds = self.get_frame_embedding(pixel_values)

        # Start with BOS token
        input_ids = torch.full((pixel_values.size(0), 1), tokenizer.bos_token_id, device=device)

        inputs_embeds = self.language_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([image_embeds.unsqueeze(1), inputs_embeds], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
        )
        return outputs

# === Load Tokenizer and Models ===
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
language_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# === Initialize and Load Model ===
model = ImagePrefixCaptioner(
    vision_model=vision_model,
    language_model=language_model,
    proj_dim=language_model.config.n_embd
).to(device)
model.load_state_dict(torch.load("captioning_model.pt", map_location=device))
model.eval()

# === Load Test Data ===
_, test_loader = get_loaders(tokenizer)

# === Evaluation Metrics ===
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

predictions = []
references = []

# === Inference Loop ===
for batch in tqdm(test_loader, desc="Evaluating"):
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"]

    # Generate captions
    generated_ids = model.generate_caption(pixel_values)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Ground truth
    reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    predictions.extend(generated_texts)
    references.extend(reference_texts)

# === Evaluate ===
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
