import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self,
                 input_dim_1: int,
                 input_dim_2: int,
                 output_dim: int,
                 fusion_type: str = "concat"):
        """
        fusion_type: one of ["concat", "sum", "mean", "mlp"]
        """
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.output_dim = output_dim

        if self.fusion_type == "concat":
            fusion_input_dim = input_dim_1 + input_dim_2
            self.fusion = nn.Identity()
        elif self.fusion_type in ["sum", "mean"]:
            assert input_dim_1 == input_dim_2, "Sum/mean fusion requires same input dims"
            fusion_input_dim = input_dim_1
            self.fusion = nn.Identity()
        elif self.fusion_type == "mlp":
            fusion_input_dim = input_dim_1 + input_dim_2
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_input_dim),
                nn.ReLU(),
                nn.Linear(fusion_input_dim, fusion_input_dim)
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        # Final projection to output_dim if needed
        self.project = nn.Linear(fusion_input_dim, output_dim) if fusion_input_dim != output_dim else nn.Identity()

    def forward(self, embed1, embed2):
        """
        embed1: (B, D1)
        embed2: (B, D2)
        Returns: (B, output_dim)
        """
        # if self.fusion_type == "concat" or self.fusion_type == "mlp":
        #     fused = torch.cat([embed1, embed2], dim=-1)
        # elif self.fusion_type == "sum":
        #     fused = embed1 + embed2
        # elif self.fusion_type == "mean":
        #     fused = (embed1 + embed2) / 2.0
        # else:
        #     raise ValueError("Invalid fusion type")
        # print(embed1.shape, embed2.shape)
        # return self.project(self.fusion(fused))
        return torch.cat([embed1, embed2], dim=1)