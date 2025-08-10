import torch
import torch.nn as nn 
from transformers import AutoModelForCausalLM, AutoTokenizer

class TextDecoder(nn.Module):
    def __init__(self, decoder_name: str):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.language_model = AutoModelForCausalLM.from_pretrained(decoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def forward(self,image_embeddings, captions):
        # Tokenize captions
        tokenized = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        
        # Embed text tokens
        inputs_embeds = self.language_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([image_embeddings.unsqueeze(1), inputs_embeds], dim=1)

        # Repeat image embeddings to match sequence length
        repeated_embeddings = image_embeddings.unsqueeze(1)
        prefix_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        # Forward through GPT2
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=torch.cat([torch.full((input_ids.shape[0], 1), -100, device=self.device), input_ids], dim=1)
        )
        return outputs
    def generate(self, image_embeddings, max_length=128, num_beams=4):
        # Create dummy input_ids with just the BOS token
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
        print(f"Input IDs shape: {input_ids.shape}")
        
        # Prepare the image embedding as the prefix token
        image_embeddings = image_embeddings.unsqueeze(1)  # (B, 1, hidden_dim)
        # Create dummy input_ids (start with just the image prefix)
        B = image_embeddings.size(0)
        dummy_input_ids = torch.full((B, 1), self.tokenizer.eos_token_id, device=self.device)

        # Generate captions with image embedding as prefix
        generated_ids = self.language_model.generate(
            input_ids=dummy_input_ids,
            max_length=max_length,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.eos_token_id,
            inputs_embeds=image_embeddings,
            early_stopping=True
        )
        return generated_ids
