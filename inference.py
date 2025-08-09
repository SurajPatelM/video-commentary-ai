import torch
import hydra
from omegaconf import DictConfig
from models.model import SwinBart
from PIL import Image
from torchvision import transforms
import os

@hydra.main(config_path="configs", config_name="default")
def inference(cfg: DictConfig):
    # Set device
    device = torch.device(cfg.trainer.device)

    # Initialize and load model
    model = SwinBart(cfg).to(device)
    model.load_state_dict(torch.load("/Users/vishwajeethogale/Desktop/Research/video-commentary-ai/checkpoints/best_model.pth", map_location=device))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load and prepare image
    image_path = cfg.inference.image_path
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            images=image_tensor,
            max_length=cfg.inference.max_length,
            num_beams=cfg.inference.num_beams
        )

    # Decode caption using model's tokenizer
    caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nGenerated Caption: {caption}\n")

if __name__ == "__main__":
    inference()
