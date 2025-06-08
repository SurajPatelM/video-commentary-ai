# generate_all.py
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPModel
import matplotlib.pyplot as plt
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# === Model definition ===
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

# === Image caption generation ===
def generate_caption(model, tokenizer, image_path, transform, generate_kwargs):
    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        generated_ids = model(pixel_values, generate_kwargs)
        caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# === Main execution ===
def caption_all_frames(folder_path, weights_path="captioning_model.pt", save_results=True):
    # Load model and tokenizer once
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    language_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    model = ImagePrefixCaptioner(
        vision_model, language_model, proj_dim=language_model.config.n_embd
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    generate_kwargs = {
        "max_length": 50,
        "num_beams": 5,
        "early_stopping": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    captions = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"\n📂 Processing {len(image_files)} frames from: {folder_path}\n")

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        st = time()
        try:
            caption = generate_caption(model, tokenizer, image_path, transform, generate_kwargs)
            elapsed = time() - st
            print(f"{filename}: {caption} ({elapsed:.2f}s)")
            captions.append((filename, caption))
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

    # Save results
    if save_results:
        output_file = os.path.join(folder_path, "captions.txt")
        with open(output_file, "w") as f:
            for fname, cap in captions:
                f.write(f"{fname}\t{cap}\n")
        print(f"\n✅ Captions saved to: {output_file}")

    return captions

if __name__ == "__main__":
    import sys
    st = time()
    folder = sys.argv[1] if len(sys.argv) > 1 else "frames/Basic Clinical Skills_ Nasogastric tube insertion_clip_18.0s_to_318.0s"
    caption_all_frames(folder)
    end = time()
    print(f"Total time taken: {end - st:.2f} seconds")
