import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer

# === Directories ===
CAPTION_DIR = "captions"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# === Load Phi-2 ===
phi_model_name = "microsoft/phi-2"
phi_tokenizer = AutoTokenizer.from_pretrained(phi_model_name)
phi_model = AutoModelForCausalLM.from_pretrained(phi_model_name, torch_dtype=torch.float32)
phi_model.eval()

# === Load Flan-T5 for summarization ===
summ_model_name = "google/flan-t5-base"
summ_tokenizer = T5Tokenizer.from_pretrained(summ_model_name)
summ_model = T5ForConditionalGeneration.from_pretrained(summ_model_name)
summ_model.eval()

# === Use CPU only for safety ===
device = torch.device("cpu")
phi_model.to(device)
summ_model.to(device)

# === Summarize long captions if needed ===
def summarize_captions(captions_dict):
    full_text = " ".join(captions_dict.values())
    input_text = "summarize: " + full_text
    inputs = summ_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    summary_ids = summ_model.generate(inputs["input_ids"], max_length=256)
    summary = summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# === Generate commentary with Phi-2 ===
def generate_commentary(prompt_text):
    inputs = phi_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = phi_model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=phi_tokenizer.eos_token_id
    )
    return phi_tokenizer.decode(output[0], skip_special_tokens=True)

# === Main inference function ===
def run_inference(caption_file):
    with open(caption_file, "r") as f:
        captions = json.load(f)

    # Summarize long captions
    summary = summarize_captions(captions)

    # Create generation prompt
    prompt = (
        "Write a detailed, coherent medical commentary based on this summary of visual frames:\n"
        f"{summary}\n\nCommentary:"
    )

    commentary = generate_commentary(prompt)
    base_name = os.path.splitext(os.path.basename(caption_file))[0]

    with open(os.path.join(RESULT_DIR, f"{base_name}_commentary.txt"), "w") as f:
        f.write("Generated Commentary:\n")
        f.write(commentary)

# === Run on all caption files ===
if __name__ == "__main__":
    for caption_file in os.listdir(CAPTION_DIR):
        if caption_file.endswith(".json"):
            print(f"🧠 Generating commentary for {caption_file}")
            run_inference(os.path.join(CAPTION_DIR, caption_file))
