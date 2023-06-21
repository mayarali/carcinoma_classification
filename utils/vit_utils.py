import vision_transformer as vits
from functools import partial
import torch
import os
import torch.nn as nn
import torch.functional as F


def load_pretrained_vit(drop_rate=0.2, img_size=[896], attn_drop_rate=0):
    custom_vit = vits.VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=img_size,
        drop_rate=drop_rate,
        num_classes=0,
        attn_drop_rate=attn_drop_rate,
    )
    custom_vit = assign_pretrained_weights(custom_vit)
    return custom_vit


def assign_pretrained_weights(
    model,
    pretrained_weights="./HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth",
    checkpoint_key="teacher",
):
    if os.path.isfile(pretrained_weights):
        print(f"Loding pretrained weights from {pretrained_weights}")
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove pos embed weights to adapt for different image size
        del state_dict["pos_embed"]
        msg = model.load_state_dict(state_dict, strict=False)
        # msg = None
        print(
            f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}"
        )
        return model
    else:
        raise FileNotFoundError(
            "No checkpoint found at {}. Please specify a checkpoint_path.".format(
                pretrained_weights
            )
        )

class CustomVIT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        pass


class ClassificationHead(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=3, dropout=0.0):
        super().__init__()
        self.fc_1 = nn.Linear(in_features, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, num_classes)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout_1(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.fc_2(x)
        return x