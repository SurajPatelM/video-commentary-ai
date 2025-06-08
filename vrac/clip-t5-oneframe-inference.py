import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    CLIPModel,
)
from transformers.modeling_outputs import BaseModelOutput
from time import time

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

# === CLIP Preprocessing ===
clip_preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4815, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2758]),
])

# === Prediction Function ===
def predict_from_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    pixel_values = clip_preprocess(image).unsqueeze(0).to(device)

    # Dummy input for T5 (empty string)
    dummy_input = tokenizer(
        "", return_tensors="pt", padding="max_length", max_length=1, truncation=True
    )
    input_ids = dummy_input["input_ids"].to(device)
    attention_mask = dummy_input["attention_mask"].to(device)

    # Generate caption
    with torch.no_grad():
        generated_ids = model(pixel_values, input_ids, attention_mask)
        caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\n🖼️ Caption for '{image_path}':\n{caption}")
    return caption

# === Example Usage ===
if __name__ == "__main__":
    import sys
    st = time()
    image_path = sys.argv[1] if len(sys.argv) > 1 else "frames/NR 224 NG tube insertion and removal_clip_0.0s_to_560.0s/frame_0120.jpg"
    predict_from_image(image_path)
    end = time()
    print(f"Time taken: {end - st:.2f} seconds")
