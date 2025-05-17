import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import json
from torchvision import transforms

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
    
    
    
if __name__ == "__main__":
    image_datadicts = dict()
    root = "./captions"
    for file in os.listdir(root):
        json_path = os.path.join(root, file)

        with open(json_path, 'r') as f:
            data = json.load(f)
        image_datadicts[str(file[:-5])] = data
        
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])
    dataset = VideoDataset(image_datadicts, transform=transform)
    # Define split lengths (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    
    
