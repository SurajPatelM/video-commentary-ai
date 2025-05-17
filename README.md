# VideoCommentary AI Pipeline

This project generates detailed commentary for instructional videos using keyframe extraction, image captioning, and language models.

## Features
- Extracts keyframes from videos
- Generates frame-level captions using BLIP
- Produces coherent medical commentary using Phi-2 (LLM)
- Runs on MacBook Air (Apple Silicon)

## Folder Structure
- `videos/`: Raw input videos
- `frames/`: Extracted keyframes
- `captions/`: Captions JSONs per video
- `results/`: Final commentaries

## Usage
### Preprocessing (frames + captions)
```bash
python preprocessing.py
```

### Dataloader based on captions, frames, videos folders
```bash
python dataloader.py
```

### Train the base model
```bash
python clip-gpt2-train.py
```

### Inference of the base model
```bash
python clip-gpt2-inference.py
```

