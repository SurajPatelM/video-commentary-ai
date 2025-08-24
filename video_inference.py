import torch
import hydra
from omegaconf import DictConfig
from models.model import SwinBart
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm






def predict_and_annotate_images(cfg, model, transform, device):
    image_dir = cfg.inference.video_image_dir
    save_dir = os.path.join(cfg.inference.inf_save_dir, "captioned")
    os.makedirs(save_dir, exist_ok=True)

    valid_exts = {'.jpg', '.jpeg', '.png'}

    image_paths = sorted([
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(image_dir)
        for f in filenames
        if os.path.splitext(f)[-1].lower() in valid_exts
    ])

    print(f"\nFound {len(image_paths)} image(s) in {image_dir}\n")

    for image_path in tqdm(image_paths, desc="Generating captions"):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Load corresponding MFCC
            audio_path = image_path.replace(f"{os.sep}frames{os.sep}", f"{os.sep}mfccs{os.sep}")
            audio_path = audio_path.replace(f"{os.sep}frames_dup{os.sep}", f"{os.sep}mfccs{os.sep}")
            audio_path = os.path.splitext(audio_path)[0] + ".npy"

            if not os.path.exists(audio_path):
                print(f"MFCC file not found: {audio_path}")
                continue

            audio_arr = np.load(audio_path)
            audio_tensor = torch.from_numpy(audio_arr).float()
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3 and audio_tensor.shape[0] != 1:
                audio_tensor = audio_tensor[:1]
            audio_tensor = audio_tensor.to(device)

            # Generate caption
            with torch.no_grad():
                generated_ids = model.generate(
                    images=image_tensor,
                    audio=audio_tensor,
                    max_length=cfg.inference.max_length,
                    num_beams=cfg.inference.num_beams
                )
            caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Draw caption on image
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            margin = 10
            text_pos = (margin, annotated_image.height - 30)
            draw.rectangle([text_pos, (annotated_image.width - margin, annotated_image.height - margin)],
                           fill=(0, 0, 0, 150))
            draw.text(text_pos, caption, font=font, fill=(255, 255, 255))

            # Save annotated image
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            annotated_image.save(save_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

@hydra.main(config_path="configs", config_name="default")
def inference(cfg: DictConfig):
    # Set device
    device = torch.device(cfg.trainer.device)

    # Initialize and load model
    model = SwinBart(cfg).to(device)
    model.load_state_dict(torch.load(cfg.inference.model_path, map_location=device))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # # Load and prepare image
    # image_path = cfg.inference.image_path
    # image = Image.open(image_path).convert("RGB")
    # image_tensor = transform(image).unsqueeze(0).to(device)

    # audio_path = image_path.replace(f"{os.sep}frames{os.sep}", f"{os.sep}mfccs{os.sep}")
    # audio_path = audio_path.replace(f"{os.sep}frames_dup{os.sep}", f"{os.sep}mfccs{os.sep}")
    # audio_path = os.path.splitext(audio_path)[0] + ".npy"

    # if not os.path.exists(audio_path):
    #     raise FileNotFoundError(f"MFCC file not found: {audio_path}")

    # audio_arr = np.load(audio_path)                 # (n_mfcc, T) or (1, n_mfcc, T)
    # audio_tensor = torch.from_numpy(audio_arr).float()
    # if audio_tensor.dim() == 2:
    #     audio_tensor = audio_tensor.unsqueeze(0)    # -> (1, n_mfcc, T)
    # elif audio_tensor.dim() == 3 and audio_tensor.shape[0] != 1:
    #     audio_tensor = audio_tensor[:1]             # single sample
    # audio_tensor = audio_tensor.to(device)
    # # Generate caption
    # with torch.no_grad():
    #     generated_ids = model.generate(
    #         images=image_tensor,
    #         audio=audio_tensor,
    #         max_length=cfg.inference.max_length,
    #         num_beams=cfg.inference.num_beams
    #     )

    # # Decode caption using model's tokenizer
    # caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # print(f"\nGenerated Caption: {caption}\n")
    predict_and_annotate_images(cfg, model,transform, device)

if __name__ == "__main__":
    inference()



