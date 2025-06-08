import torch 
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

class Decoder(nn.Module):

    input_dim : int
    decoder : str
    lora : bool
    output_dim : int
    freeze : bool

    def __init__(self, 
                input_dim, 
                decoder,
                lora,
                output_dim,
                freeze):
        super().__init__()
        self.input_dim = input_dim
        self.expected_dim = None
        if decoder == "gpt2":
            self.model = AutoModelForCausalLM.from_pretrained(decoder)
            self.tokenizer = AutoTokenizer.from_pretrained(decoder)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.expected_dim = getattr(self.model.config, "n_embd", self.model.config.hidden_size)
        if lora:
            # Create LoRA config
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["c_attn"],  # GPT-2 uses 'c_attn' in attention blocks
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            # Wrap the model with PEFT
            self.model = get_peft_model(self.model, lora_config)
        self.output_dim = output_dim
        self.freeze = freeze
        if freeze:
            for name, param in self.model.named_parameters():
                if not any(n in name.lower() for n in ["lora_", "adapter"]):
                    param.requires_grad = False
        self.parameters_info = self._get_trainable_parameters()
        self.linear_projection = nn.Linear(input_dim, self.expected_dim) if input_dim != self.expected_dim else nn.Identity()

    def _get_trainable_parameters(self):
        param_info = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_info.append({
                    "name": name,
                    "shape": tuple(param.shape),
                    "numel": param.numel()
                })
        return param_info

    def forward(self, x, attention_mask=None, labels=None):
        projected = self.linear_projection(x)
        outputs = self.model(inputs_embeds=projected, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits
