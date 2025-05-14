import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

CAPTION_DIR = "captions"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def generate_commentary(captions_dict):
    prompt = "Create a detailed medical commentary based on these image descriptions:\n"
    for img, desc in captions_dict.items():
        prompt += f"{img}: {desc}\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=600)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def run_inference(caption_file):
    with open(caption_file, "r") as f:
        captions = json.load(f)
    
    commentary = generate_commentary(captions)
    base_name = os.path.splitext(os.path.basename(caption_file))[0]
    
    with open(os.path.join(RESULT_DIR, f"{base_name}_commentary.txt"), "w") as f:
        f.write("Generated Commentary:\n")
        f.write(commentary)

if __name__ == "__main__":
    for caption_file in os.listdir(CAPTION_DIR):
        print(f"Generating commentary for {caption_file}")
        run_inference(os.path.join(CAPTION_DIR, caption_file))
