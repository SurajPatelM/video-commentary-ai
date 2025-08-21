import os
import cv2
import glob
import json
import numpy as np
import subprocess
from PIL import Image
import torch
import librosa
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Directories
VIDEO_DIR = "videos"
FRAME_DIR = "frames_dup"
MFCC_DIR = "mfccs"
CAPTION_DIR = "captions"
TEMP_DIR = "temp"

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(MFCC_DIR, exist_ok=True)
os.makedirs(CAPTION_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

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
    return fps

def extract_audio_with_ffmpeg(video_path, output_audio_path):
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if exists
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", "44100",  # Sample rate
        "-ac", "1",  # Mono audio
        output_audio_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def compute_mfccs(
    audio_path,
    num_frames,
    output_folder,
    sr_target=44100,
    n_mfcc=13,
    n_fft=1024,
    hop_length=256,
    n_mels=40,
    cmvn=True,
):
    """
    Slice the whole audio evenly into `num_frames` segments (aligned to the
    number of saved JPGs) and compute MFCC for each slice.

    - Does NOT change frame extraction.
    - Robust even when video FPS varies or when exact frame timestamps drift.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load audio (match ffmpeg settings; librosa handles resample if needed)
    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    total = len(y)

    if num_frames == 0 or total == 0:
        return

    # Segment boundaries: cover the entire signal with equal parts
    # e.g., for N segments we create N+1 cut points from 0..total
    cuts = np.linspace(0, total, num_frames + 1, dtype=np.int64)

    for i in range(num_frames):
        start = int(cuts[i])
        end = int(cuts[i + 1])
        seg = y[start:end]

        # Handle edge cases (very short or empty slices)
        if seg.size == 0:
            # Save a tiny zero patch so downstream code finds a file
            seg = np.zeros(hop_length, dtype=np.float32)

        # MFCCs (explicit params for stability across varying segment lengths)
        mfcc = librosa.feature.mfcc(
            y=seg, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )  # shape: (n_mfcc, T_seg)

        # Optional per-channel CMVN to stabilize scale across segments
        if cmvn:
            mu = mfcc.mean(axis=1, keepdims=True)
            sd = mfcc.std(axis=1, keepdims=True) + 1e-5
            mfcc = (mfcc - mu) / sd

        out_path = os.path.join(output_folder, f"frame_{i:04d}.npy")
        np.save(out_path, mfcc)


def process_video(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(FRAME_DIR, name)
    mfcc_folder = os.path.join(MFCC_DIR, name)
    audio_path = os.path.join(TEMP_DIR, f"{name}.wav")

    print(f"\n=== Processing {video_path} ===")
    print(f"Extracting frames...")
    fps = extract_key_frames(video_path, frame_folder)     # UNCHANGED
    num_frames = len(glob.glob(f"{frame_folder}/*.jpg"))

    print(f"Extracting audio with ffmpeg...")
    extract_audio_with_ffmpeg(video_path, audio_path)      # UNCHANGED

    print(f"Computing MFCCs...")
    compute_mfccs(audio_path, num_frames, mfcc_folder)     # <-- only this call changed


if __name__ == "__main__":
    for video in glob.glob(os.path.join(VIDEO_DIR, "*.mp4")):
        process_video(video)
