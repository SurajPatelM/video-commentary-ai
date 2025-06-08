import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from peft import get_peft_model, LoraConfig, TaskType

class Encoder(nn.Module):
    def __init__(self,
                 input_dim:int,
                 encoder: str,
                 lora: bool = False,
                 output_dim: int = 768,
                 freeze: bool = False):
        super().__init__()

        self.encoder_name = encoder
        self.lora = lora
        self.output_dim = output_dim
        self.freeze = freeze

        if "vit" in encoder.lower():
            self.modality = "vision"
            self.processor = AutoFeatureExtractor.from_pretrained(encoder)
            task_type = TaskType.FEATURE_EXTRACTION
            target_modules = ["query", "value"]
        else:
            self.modality = "text"
            self.processor = AutoTokenizer.from_pretrained(encoder)
            task_type = TaskType.FEATURE_EXTRACTION  # or SEQ_CLS for classification
            target_modules = ["query", "value", "key"]

        self.model = AutoModel.from_pretrained(encoder)
        self.hidden_dim = self.model.config.hidden_size

        if lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=task_type
            )
            self.model = get_peft_model(self.model, lora_config)

        if freeze:
            for name, param in self.model.named_parameters():
                # Freeze all except LoRA layers
                if not any(n in name.lower() for n in ["lora_", "adapter"]):
                    param.requires_grad = False

        self.project = nn.Linear(self.hidden_dim, output_dim) if self.hidden_dim != output_dim else nn.Identity()

        self.parameters_info = self._get_trainable_parameters()

    def _get_trainable_parameters(self):
        return [
            {
                "name": name,
                "shape": tuple(param.shape),
                "numel": param.numel()
            }
            for name, param in self.named_parameters() if param.requires_grad
        ]

    def forward(self, inputs):
        if self.modality == "vision":
            outputs = self.model(pixel_values=inputs)
            cls_embedding = outputs.last_hidden_state  # [CLS] token
        else:  # text
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state
        return self.project(cls_embedding)
