# generate.py
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


# === Model definition (same as train.py) ===
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


def generate_caption(image_path, weights_path="captioning_model.pt"):
    # === Load tokenizer and models ===
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    language_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # === Build model and load weights ===
    model = ImagePrefixCaptioner(
        vision_model=vision_model,
        language_model=language_model,
        proj_dim=language_model.config.n_embd
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # === Preprocess image ===
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
        print(caption)
    return caption, image


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "frames/NR 224 NG tube insertion and removal_clip_0.0s_to_560.0s/frame_0020.jpg"
    caption, image = generate_caption(image_path)

    # === Display image and caption ===
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"🖼️  Caption: {caption}", fontsize=12)
    plt.tight_layout()
    plt.show()
