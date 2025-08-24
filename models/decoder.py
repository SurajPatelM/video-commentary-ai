from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn 
import torch

class TextDecoder(nn.Module):
    def __init__(self, decoder_name: str):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
        self.bart_model = BartForConditionalGeneration.from_pretrained(decoder_name).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(decoder_name)
        self.projection_images_audio = nn.Linear(1536, 768)
    def forward(self, image_audio_embeddings, captions):
        # Tokenize captions
        tokenized = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=128)

        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        seq_len = input_ids.shape[1]
        # print(f"Image/Audio embeddings shape: {image_audio_embeddings.shape}")

        # Project image/audio embeddings to match BART's hidden size
        image_audio_embeddings = self.projection_images_audio(image_audio_embeddings)
        # print(f"Image/Audio embeddings shape: {image_audio_embeddings.shape}")
        # Repeat image embeddings to match sequence length
        repeated_embeddings = image_audio_embeddings.unsqueeze(1).repeat(1, seq_len, 1).to(self.device)

        # Forward through BART
        outputs = self.bart_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            encoder_outputs=(repeated_embeddings, ),
        )
        return outputs
    def generate(self, image_audio_embeddings, max_length=128, num_beams=4):
        # Create dummy input_ids with just the BOS token
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
        print(f"Input IDs shape: {input_ids.shape}")
        # BART expects `encoder_outputs` to be a model output object, not just a tuple
        # encoder_outputs = image_embeddings.unsqueeze(1).to(self.device)
        encoder_hidden_states = self.projection_images_audio(image_audio_embeddings)
        print(encoder_hidden_states.shape)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).to(self.device)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        print(f"Encoder outputs shape: {encoder_hidden_states.shape}")
        
        generated_ids = self.bart_model.generate(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        return generated_ids
