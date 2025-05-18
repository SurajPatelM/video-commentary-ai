import os
import cv2
import glob
import json
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Directories
VIDEO_DIR = "videos"
FRAME_DIR = "frames"
CAPTION_DIR = "captions"
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(CAPTION_DIR, exist_ok=True)

# Load model and processors
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to extract key frames from a video
def extract_key_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(1)) % int(fps * frame_rate) == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1
    cap.release()

# Function to generate caption for an image
def caption_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Function to process a video and save frame captions
def process_video(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(FRAME_DIR, name)
    extract_key_frames(video_path, frame_folder)

    captions = {}
    for frame in sorted(glob.glob(f"{frame_folder}/*.jpg")):
        caption = caption_image(frame)
        captions[os.path.basename(frame)] = caption

    with open(os.path.join(CAPTION_DIR, f"{name}.json"), "w") as f:
        json.dump(captions, f, indent=2)

# Main loop to process all videos
if __name__ == "__main__":
    for video in glob.glob(os.path.join(VIDEO_DIR, "*.mp4")):
        print(f"Processing {video}")
        process_video(video)
