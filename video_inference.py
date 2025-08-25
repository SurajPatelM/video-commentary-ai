# import torch
# import hydra
# from omegaconf import DictConfig
# from models.model import SwinBart
# from PIL import Image, ImageDraw, ImageFont
# from torchvision import transforms
# import os
# import pandas as pd
# import numpy as np
# from tqdm import tqdm


# def predict_and_annotate_images(cfg, model, transform, device):
#     image_dir = cfg.inference.video_image_dir
#     save_dir = os.path.join(cfg.inference.inf_save_dir, "captioned")
#     os.makedirs(save_dir, exist_ok=True)

#     ground_truth_captions_csv = pd.read_csv(os.path.join(cfg.data.captions_dir, image_dir.split("/")[-1] + ".csv"))
#     ground_truth_captions = ground_truth_captions_csv["caption"].tolist()
#     predictions = []
#     valid_exts = {'.jpg', '.jpeg', '.png'}

#     image_paths = sorted([
#         os.path.join(dp, f)
#         for dp, dn, filenames in os.walk(image_dir)
#         for f in filenames
#         if os.path.splitext(f)[-1].lower() in valid_exts
#     ])

#     print(f"\nFound {len(image_paths)} image(s) in {image_dir}\n")

#     for image_path in tqdm(image_paths, desc="Generating captions"):
#         try:
#             image = Image.open(image_path).convert("RGB")
#             image_tensor = transform(image).unsqueeze(0).to(device)

#             # Load corresponding MFCC
#             audio_path = image_path.replace(f"{os.sep}frames{os.sep}", f"{os.sep}mfccs{os.sep}")
#             audio_path = audio_path.replace(f"{os.sep}frames_dup{os.sep}", f"{os.sep}mfccs{os.sep}")
#             audio_path = os.path.splitext(audio_path)[0] + ".npy"

#             if not os.path.exists(audio_path):
#                 print(f"MFCC file not found: {audio_path}")
#                 continue

#             audio_arr = np.load(audio_path)
#             audio_tensor = torch.from_numpy(audio_arr).float()
#             if audio_tensor.dim() == 2:
#                 audio_tensor = audio_tensor.unsqueeze(0)
#             elif audio_tensor.dim() == 3 and audio_tensor.shape[0] != 1:
#                 audio_tensor = audio_tensor[:1]
#             audio_tensor = audio_tensor.to(device)

#             # Generate caption
#             with torch.no_grad():
#                 generated_ids = model.generate(
#                     images=image_tensor,
#                     audio=audio_tensor,
#                     max_length=cfg.inference.max_length,
#                     num_beams=cfg.inference.num_beams
#                 )
#             caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#             predictions.append(caption)
            
#             # Draw caption on image
#             annotated_image = image.copy()
#             draw = ImageDraw.Draw(annotated_image)
#             try:
#                 font = ImageFont.truetype("arial.ttf", 20)
#             except IOError:
#                 font = ImageFont.load_default()

#             margin = 10
#             text_pos = (margin, annotated_image.height - 30)
#             draw.rectangle([text_pos, (annotated_image.width - margin, annotated_image.height - margin)],
#                            fill=(0, 0, 0, 150))
#             draw.text(text_pos, caption, font=font, fill=(255, 255, 255))

#             # Save annotated image
#             save_path = os.path.join(save_dir, os.path.basename(image_path))
#             annotated_image.save(save_path)

#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")

# @hydra.main(config_path="configs", config_name="default")
# def inference(cfg: DictConfig):
#     # Set device
#     device = torch.device(cfg.trainer.device)

#     # Initialize and load model
#     model = SwinBart(cfg).to(device)
#     model.load_state_dict(torch.load(cfg.inference.model_path, map_location=device))
#     model.eval()

#     # Image preprocessing
#     transform = transforms.Compose([
#         transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     predict_and_annotate_images(cfg, model,transform, device)

# if __name__ == "__main__":
#     inference()



import torch
import hydra
from omegaconf import DictConfig
from models.model import SwinBart
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- NEW: imports for eval ---
import re
import json
from collections import Counter
import math

# --- NEW: text + metric utilities ---
def _normalize_text(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[\u2018\u2019]", "'", s)
    return s

def _tok(s: str):
    return _normalize_text(s).split()

def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def _corpus_bleu(hypotheses_tokens, references_tokens_list, max_n=4):
    """
    Minimal BLEU with add-1 smoothing and (single/multi)-reference support.
    hypotheses_tokens: list[list[str]]
    references_tokens_list: list[list[list[str]]]  (multi-refs per item)
    """
    precisions = []
    for n in range(1, max_n + 1):
        overlap = 0
        total = 0
        for hyp, refs in zip(hypotheses_tokens, references_tokens_list):
            hyp_ngrams = _ngram_counts(hyp, n)
            max_ref_counts = Counter()
            for r in refs:
                ref_ngrams = _ngram_counts(r, n)
                for ng, cnt in ref_ngrams.items():
                    if cnt > max_ref_counts[ng]:
                        max_ref_counts[ng] = cnt
            for ng, cnt in hyp_ngrams.items():
                overlap += min(cnt, max_ref_counts.get(ng, 0))
            total += sum(hyp_ngrams.values())
        precisions.append((overlap + 1) / (total + 1))  # add-1 smoothing

    hyp_len_total = sum(len(h) for h in hypotheses_tokens)
    ref_len_total = 0
    for hyp, refs in zip(hypotheses_tokens, references_tokens_list):
        hyp_len = len(hyp)
        # closest reference length
        closest = min((abs(len(r) - hyp_len), len(r)) for r in refs)
        ref_len_total += closest[1]

    if hyp_len_total == 0:
        bp = 0.0
    elif hyp_len_total > ref_len_total:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len_total / hyp_len_total)

    log_prec = sum(math.log(p) for p in precisions) / max_n
    bleu = bp * math.exp(log_prec)
    return bleu, bp, precisions

def _lcs_len(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        ai = a[i-1]
        row_i = dp[i]
        row_im1 = dp[i-1]
        for j in range(1, n+1):
            if ai == b[j-1]:
                row_i[j] = row_im1[j-1] + 1
            else:
                row_i[j] = max(row_im1[j], row_i[j-1])
    return dp[m][n]

def _rouge_l_f1(hyp_tokens, ref_tokens):
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(hyp_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    p = lcs / len(hyp_tokens)
    r = lcs / len(ref_tokens)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

# --- NEW: parse frame index from filename 'frame_0007.jpg' -> 7 ---
def _parse_frame_idx(filename: str):
    base = os.path.basename(filename)
    m = re.search(r'frame[_-]?(\d+)', base, flags=re.IGNORECASE)
    if not m:
        # fallback: last digit-run in the name
        m2 = re.findall(r'(\d+)', base)
        if not m2:
            return None
        return int(m2[-1])
    return int(m.group(1))

def predict_and_annotate_images(cfg, model, transform, device):
    image_dir = cfg.inference.video_image_dir
    save_dir = os.path.join(cfg.inference.inf_save_dir, "captioned")
    os.makedirs(save_dir, exist_ok=True)

    # Ground-truth loaded and mapped by row index (row 0 -> frame_0000.jpg)
    gt_csv_path = os.path.join(cfg.data.captions_dir, image_dir.split("/")[-1] + ".csv")
    ground_truth_captions_csv = pd.read_csv(gt_csv_path)
    if "caption" not in ground_truth_captions_csv.columns:
        raise ValueError("Ground truth CSV must contain a 'caption' column.")
    ground_truth_captions = ground_truth_captions_csv["caption"].tolist()

    predictions = []
    pred_rows = []  # for saving and eval: frame_idx, image, prediction, reference (if available)
    valid_exts = {'.jpg', '.jpeg', '.png'}

    image_paths = sorted([
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(image_dir)
        for f in filenames
        if os.path.splitext(f)[-1].lower() in valid_exts
    ])

    print(f"\nFound {len(image_paths)} image(s) in {image_dir}\n")

    for image_path in tqdm(image_paths, desc="Generating captions"):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Load corresponding MFCC
            audio_path = image_path.replace(f"{os.sep}frames{os.sep}", f"{os.sep}mfccs{os.sep}")
            audio_path = audio_path.replace(f"{os.sep}frames_dup{os.sep}", f"{os.sep}mfccs{os.sep}")
            audio_path = os.path.splitext(audio_path)[0] + ".npy"

            if not os.path.exists(audio_path):
                print(f"MFCC file not found: {audio_path}")
                continue

            audio_arr = np.load(audio_path)
            audio_tensor = torch.from_numpy(audio_arr).float()
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3 and audio_tensor.shape[0] != 1:
                audio_tensor = audio_tensor[:1]
            audio_tensor = audio_tensor.to(device)

            # Generate caption
            with torch.no_grad():
                generated_ids = model.generate(
                    images=image_tensor,
                    audio=audio_tensor,
                    max_length=cfg.inference.max_length,
                    num_beams=cfg.inference.num_beams
                )
            caption = model.decoder.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            predictions.append(caption)

            # --- record with frame index + reference by row index ---
            img_name = os.path.basename(image_path)
            frame_idx = _parse_frame_idx(img_name)
            ref = ground_truth_captions[frame_idx] if (frame_idx is not None and frame_idx < len(ground_truth_captions)) else ""
            pred_rows.append({
                "frame_idx": frame_idx,
                "image": img_name,
                "prediction": caption,
                "reference": ref
            })

            # Draw caption on image
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            margin = 10
            text_pos = (margin, annotated_image.height - 30)
            draw.rectangle([text_pos, (annotated_image.width - margin, annotated_image.height - margin)],
                           fill=(0, 0, 0, 150))
            draw.text(text_pos, caption, font=font, fill=(255, 255, 255))

            # Save annotated image
            save_path = os.path.join(save_dir, img_name)
            annotated_image.save(save_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # ===== EVALUATION (BLEU-4 + ROUGE-L F1) =====
    eval_dir = os.path.join(cfg.inference.inf_save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Save predictions table with references
    pred_df = pd.DataFrame(pred_rows)
    pred_csv_path = os.path.join(eval_dir, "predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    # Filter rows that have a valid frame_idx and in-range reference
    valid_eval = pred_df.dropna(subset=["frame_idx"]).astype({"frame_idx": int})
    valid_eval = valid_eval[valid_eval["frame_idx"] < len(ground_truth_captions)]
    if valid_eval.empty:
        print("No valid predictions matched to ground-truth rows for evaluation.")
        metrics = {
            "matched": 0,
            "bleu": 0.0,
            "bleu_precisions": [0, 0, 0, 0],
            "brevity_penalty": 0.0,
            "rougeL_F1": 0.0
        }
    else:
        hyps = [_tok(h) for h in valid_eval["prediction"].tolist()]
        refs = [[_tok(r)] for r in valid_eval["reference"].tolist()]  # single-ref per row

        bleu, bp, precisions = _corpus_bleu(hyps, refs, max_n=4)

        # ROUGE-L: average of per-example best-ref F1 (single ref here)
        rouge_sum = 0.0
        for h_tokens, r_tokens in zip(hyps, (r[0] for r in refs)):
            rouge_sum += _rouge_l_f1(h_tokens, r_tokens)
        rouge_avg = rouge_sum / len(hyps)

        metrics = {
            "matched": int(len(hyps)),
            "bleu": float(bleu),
            "bleu_precisions": [float(p) for p in precisions],  # approx BLEU-1..4
            "brevity_penalty": float(bp),
            "rougeL_F1": float(rouge_avg)
        }

    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Evaluation Summary (row-index mapping) ===")
    print(f"Matched examples: {metrics['matched']}")
    print(f"BLEU-4: {metrics['bleu']:.4f}  (BP={metrics['brevity_penalty']:.3f}, p1..p4={['%.3f'%p for p in metrics['bleu_precisions']]})")
    print(f"ROUGE-L F1 (avg): {metrics['rougeL_F1']:.4f}")
    print(f"Saved predictions to: {pred_csv_path}")
    print(f"Saved metrics to: {metrics_path}\n")

@hydra.main(config_path="configs", config_name="default")
def inference(cfg: DictConfig):
    # Set device
    device = torch.device(cfg.trainer.device)

    # Initialize and load model
    model = SwinBart(cfg).to(device)
    model.load_state_dict(torch.load(cfg.inference.model_path, map_location=device))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((cfg.inference.image_size, cfg.inference.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    predict_and_annotate_images(cfg, model, transform, device)

if __name__ == "__main__":
    inference()
