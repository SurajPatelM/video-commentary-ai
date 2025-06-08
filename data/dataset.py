import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VideoCaptionDataset(Dataset):
    def __init__(self, captions_dir, frames_dir, transform=None):
        """
        Args:
            captions_dir (str): Path to the directory with JSON caption files.
            frames_dir (str): Path to the directory with frame subdirectories.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.captions_dir = captions_dir
        self.frames_dir = frames_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.data = [] 

        for json_file in os.listdir(captions_dir):
            if not json_file.endswith(".json"):
                continue
            video_name = os.path.splitext(json_file)[0]
            json_path = os.path.join(captions_dir, json_file)

            with open(json_path, "r") as f:
                frame_caption_map = json.load(f)

            for frame_id, caption in frame_caption_map.items():
                image_path = os.path.join(frames_dir, video_name, f"{frame_id}")
                if os.path.exists(image_path):
                    self.data.append((image_path, caption))
                else:
                    print(f"Warning: Missing image {image_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, caption
