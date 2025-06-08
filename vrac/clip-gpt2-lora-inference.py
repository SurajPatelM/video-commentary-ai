# generate.py
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPModel
from peft import PeftModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# === Same model class as in train.py ===
class ImagePrefixCaptioner(nn.Module):
    def __init__(self, vision_model, language_model, proj_dim):
        super().__init__()
        self.vision_encoder = vision_model.vision_model.eval()
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.language_model = language_model
        self.proj = nn.Linear(self.vision_encoder.config.hidden_size, proj_dim)

    def forward(self, pixel_values, generate_kwargs=None):
        with torch.no_grad():
            image_features = self.vision_encoder(pixel_values).last_hidden_state[:, 0, :]
        image_embeds = self.proj(image_features)
        prefix_embed = image_embeds.unsqueeze(1)

        if generate_kwargs:
            outputs = self.language_model.generate(
                inputs_embeds=prefix_embed,
                **generate_kwargs
            )
            return outputs
        else:
            return prefix_embed


def generate_caption(image_path, lora_path="caption_lora"):
    # === Load tokenizer and base decoder ===
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    base_decoder = AutoModelForCausalLM.from_pretrained("distilgpt2")
    language_model = PeftModel.from_pretrained(base_decoder, lora_path).to(device)

    vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    model = ImagePrefixCaptioner(
        vision_model=vision_model,
        language_model=language_model,
        proj_dim=language_model.base_model.model.config.n_embd  # GPT-2 hidden size
    ).to(device)

    model.eval()

    # === Load and transform image ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device)

    # === Generate caption ===
    generate_kwargs = {
        "max_length": 50,
        "num_beams": 5,
        "early_stopping": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    with torch.no_grad():
        generated_ids = model(pixel_values, generate_kwargs)
        caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return caption, image


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "frames/Painless Nasogastric Tube Placement_clip_24.0s_to_522.0s/frame_0100.jpg"
    caption, image = generate_caption(image_path)

    # === Display image with caption ===
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"🖼️  {caption}", fontsize=12)
    plt.tight_layout()
    plt.show()
