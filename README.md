# VideoCommentary AI Pipeline

This project generates detailed commentary for instructional videos using keyframe extraction, image captioning, and large language models. It integrates PEFT techniques, vision encoders, fusion mechanisms, and text decoders to enhance the quality of generated commentaries.

---

## Features

* Extracts keyframes from input videos
* Generates frame-level captions using BLIP
* Produces coherent medical commentary using Phi-2 (LLM)
* Integrates PEFT (e.g., LoRA) for efficient fine-tuning
* Combines visual and textual embeddings via fusion mechanisms
* Optimized to run on MacBook Air (Apple Silicon)

---

## Folder Structure

```
project-root/
├── videos/         # Raw input videos
├── frames/         # Extracted keyframes
├── captions/       # Generated captions per frame (JSON)
├── results/        # Final generated commentaries
├── configs/        # YAML configuration files
├── models/         # Pretrained and fine-tuned models
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install Dependencies

Ensure you have Python 3.8+ and run:

```bash
pip install -r requirements.txt
```

---

## Usage

### Preprocessing: Extract Frames and Generate Captions

Extract keyframes and generate image captions using BLIP:

```bash
python preprocess.py --videos_dir videos/ --frames_dir frames/ --captions_dir captions/
```

### Data Preparation

Prepare training data by aligning frames and captions:

```bash
python dataloader.py --frames_dir frames/ --captions_dir captions/
```

### Model Training

Train the base model with PEFT (LoRA) integration:

```bash
python train.py --config configs/default.yaml
```

### Inference

Generate commentary for new videos:

```bash
python infer.py --video path/to/video.mp4 --output_dir results/
```

---

## Configuration Details

Located in `configs/default.yaml`

### Decoder Configuration

```yaml
decoder: gpt2
lora: true
input_dim: 768
output_dim: 768
freeze: false
```

### Trainer Configuration

```yaml
precision: 32
accelerator: auto
devices: 2
batch_size: 4
lr: 1e-5
device: mps
epochs: 5
```

### Data Configuration

```yaml
captions_dir: captions/
frames_dir: frames/
batch_size: 8
num_workers: 4
pin_memory: true
```

---

## Research Goals

* Investigate the impact of PEFT (LoRA) on decoder performance
* Compare fusion strategies for vision-language embeddings
* Evaluate different vision encoders for frame-level captioning

---

## Results and Observations

To be updated as experiments progress.

---

## Future Work

* Explore attention-based fusion mechanisms
* Experiment with other PEFT methods (e.g., prefix tuning)
* Extend pipeline to support full video sequence modeling

---

## License

This project is licensed under the MIT License.
