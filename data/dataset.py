import os
import json
import pandas as pd
import torch
import numpy as np
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


class VideoCaptionDatasetCSV(Dataset):
    def __init__(self, captions_dir, frames_dir, audio_dir, transform=None):
        self.captions_dir = captions_dir
        self.frames_dir = frames_dir
        self.audio_dir = audio_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data = [] 

        for csv_file in os.listdir(captions_dir):
            if not csv_file.endswith(".csv"):
                continue
            video_name = os.path.splitext(csv_file)[0]
            csv_path = os.path.join(captions_dir, csv_file)
            df = pd.read_csv(csv_path)

            for frame_num, row in df.iterrows():
                frame_id = f"frame_{str(frame_num).zfill(4)}.jpg"
                caption = row['caption']
                image_path = os.path.join(frames_dir, video_name, frame_id)

                audio_path = os.path.join(audio_dir, video_name, f"frame_{str(frame_num).zfill(4)}.npy")

                if not os.path.exists(audio_path):
                    print(f"Warning: Missing audio {audio_path}")
                if os.path.exists(image_path):
                    self.data.append((image_path, audio_path, caption))
                else:
                    print(f"Warning: Missing image {image_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, audio_path, caption = self.data[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # audio
        audio_tensor = torch.tensor(np.load(audio_path), dtype=torch.float32)  # (n_mfcc, T)
        return {"images": image, "captions": caption, "audio": audio_tensor}

    @staticmethod
    def collate_fn(batch):
        images = [b["images"] for b in batch]
        captions = [b["captions"] for b in batch]
        audios = [b["audio"] for b in batch]

        # Stack images
        images_tensor = torch.stack(images)

        # Pad audio sequences
        max_T = max(a.shape[1] for a in audios)  # find longest time dimension
        n_mfcc = audios[0].shape[0]
        B = len(audios)
        audio_tensor = torch.zeros(B, n_mfcc, max_T, dtype=audios[0].dtype)

        audio_lengths = []
        for i, a in enumerate(audios):
            T = a.shape[1]
            audio_tensor[i, :, :T] = a
            audio_lengths.append(T)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        return {
            "images": images_tensor,
            "captions": captions,
            "audio": audio_tensor,         # (B, n_mfcc, T_max)
            "audio_lengths": audio_lengths # (B,) actual lengths
        }