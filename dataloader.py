import os
from PIL import Image
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, image_datadicts, transform=None):
        self.data_paths = []
        self.captions = []
        self.transform = transform
        
        for video_name, frames in image_datadicts.items():
            for frame_file, caption in frames.items():
                frame_path = os.path.join("frames", video_name, frame_file)
                self.data_paths.append(frame_path)
                self.captions.append(caption)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path = self.data_paths[idx]
        caption = self.captions[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption
    
    
    
