import os
import cv2
import glob
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

VIDEO_DIR = "videos"
FRAME_DIR = "frames"
CAPTION_DIR = "captions"
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(CAPTION_DIR, exist_ok=True)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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

def caption_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

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

if __name__ == "__main__":
    for video in glob.glob(os.path.join(VIDEO_DIR, "*.mp4")):
        print(f"Processing {video}")
        process_video(video)
