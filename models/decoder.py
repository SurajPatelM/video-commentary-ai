from transformers import BartForConditionalGeneration, BartTokenizer
import torch.nn as nn 
import torch

class TextDecoder(nn.Module):
    def __init__(self, decoder_name: str):
        super().__init__()
        self.bart_model = BartForConditionalGeneration.from_pretrained(decoder_name)
        self.tokenizer = BartTokenizer.from_pretrained(decoder_name)

    def forward(self, image_embeddings, captions):
        # Tokenize captions
        tokenized = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=128)

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        seq_len = input_ids.shape[1]

        # Repeat image embeddings to match sequence length
        repeated_embeddings = image_embeddings.unsqueeze(1).repeat(1, seq_len, 1)

        # Forward through BART
        outputs = self.bart_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            encoder_outputs=(repeated_embeddings, ),
        )
        return outputs
    def generate(self, image_embeddings, max_length=30, num_beams=4):
        # Create dummy input_ids with just the BOS token
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]])

        # BART expects `encoder_outputs` to be a model output object, not just a tuple
        encoder_outputs = image_embeddings.unsqueeze(1)
        
        generated_ids = self.bart_model.generate(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        return generated_ids
