import os
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VideoCaptionDatasetCSV(Dataset):
    def __init__(self, captions_dir, frames_dir, transform=None, csv_files=None):
        """
        Args:
            captions_dir (str): Directory with CSV caption files (one per video).
            frames_dir (str): Directory with frame subfolders (one per video).
            transform (callable, optional): Transform on image tensors.
            csv_files (list[str] | None): Optional list of CSV basenames to load (e.g., ["video1.csv"]).
        """
        self.captions_dir = captions_dir
        self.frames_dir = frames_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data = []
        files = csv_files if csv_files is not None else [
            f for f in os.listdir(captions_dir) if f.endswith(".csv")
        ]
        for csv_file in files:
            video_name = os.path.splitext(csv_file)[0]
            csv_path = os.path.join(captions_dir, csv_file)
            df = pd.read_csv(csv_path)
            for frame_num, row in df.iterrows():
                frame_id = f"frame_{str(frame_num).zfill(4)}.jpg"
                caption = row["caption"]
                image_path = os.path.join(frames_dir, video_name, frame_id)
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
        return {"images": image, "captions": caption}

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b["images"] for b in batch])
        captions = [b["captions"] for b in batch]
        return {"images": images, "captions": captions}


class VideoCaptionDataset(Dataset):
    def __init__(self, captions_dir, frames_dir, transform=None):
        """
        Args:
            captions_dir (str): Directory with JSON caption files (one per video).
            frames_dir (str): Directory with frame subfolders (one per video).
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
                image_path = os.path.join(frames_dir, video_name, frame_id)
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
        return {"images": image, "captions": caption}

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b["images"] for b in batch])
        captions = [b["captions"] for b in batch]
        return {"images": images, "captions": captions}
