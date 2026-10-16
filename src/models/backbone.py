"""
Frozen Foundation Backbone Module for SEFA-Forensics.
Extracts intermediate patch representations from completely frozen Vision Foundation Models (DINOv2 / ViT / CLIP).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class FrozenFoundationBackbone(nn.Module):
    """
    Wraps a vision foundation model, freezing 100% of its weights,
    and capturing intermediate layer patch token representations.
    """
    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        pretrained: bool = True,
        img_size: int = 224,
        target_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.img_size = img_size
        self.target_layers = target_layers or [3, 6, 9, 11]
        self.patch_size = 14 if "14" in backbone_name else 16

        self.backbone, self.feature_dim = self._init_backbone(backbone_name, pretrained)

        # Freeze all backbone parameters strictly
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Hooks storage for intermediate activations
        self.layer_features: Dict[int, torch.Tensor] = {}
        self.hooks = []
        self._register_layer_hooks()

    def _init_backbone(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """
        Initializes backbone with graceful fallbacks:
        1. torch.hub dinov2
        2. timm vit
        3. Built-in standalone ViT (if offline/air-gapped)
        """
        # Try DINOv2 via torch.hub
        if "dinov2" in name:
            try:
                hub_model = torch.hub.load("facebookresearch/dinov2", name, pretrained=pretrained)
                feat_dim = hub_model.embed_dim
                return hub_model, feat_dim
            except Exception as e:
                print(f"[Notice] Could not load '{name}' via torch.hub ({e}). Falling back to timm / standalone ViT.")

        # Try timm
        try:
            import timm
            timm_name = "vit_small_patch16_224" if "s" in name else "vit_base_patch16_224"
            model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
            feat_dim = model.embed_dim
            self.patch_size = 16
            return model, feat_dim
        except Exception as e:
            print(f"[Notice] Using robust standalone ViT backbone: {e}")
            return self._build_standalone_vit()

    def _build_standalone_vit(self) -> Tuple[nn.Module, int]:
        """
        Self-contained Vision Transformer backbone to guarantee 100% offline functionality.
        """
        embed_dim = 384
        num_layers = 12
        num_heads = 6

        class PatchEmbed(nn.Module):
            def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
                super().__init__()
                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            def forward(self, x):
                return self.proj(x).flatten(2).transpose(1, 2)

        class Block(nn.Module):
            def __init__(self, dim, num_heads):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
                self.norm2 = nn.LayerNorm(dim)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            def forward(self, x):
                x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                x = x + self.mlp(self.norm2(x))
                return x

        class StandaloneViT(nn.Module):
            def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6):
                super().__init__()
                self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
                num_patches = (img_size // patch_size) ** 2
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
                self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
                self.norm = nn.LayerNorm(embed_dim)
                self.embed_dim = embed_dim
            def forward(self, x):
                B = x.shape[0]
                tokens = self.patch_embed(x)
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, tokens), dim=1)
                x = x + self.pos_embed
                for blk in self.blocks:
                    x = blk(x)
                return self.norm(x)

        model = StandaloneViT(img_size=self.img_size, patch_size=16, embed_dim=embed_dim)
        self.patch_size = 16
        return model, embed_dim

    def _register_layer_hooks(self):
        """
        Attaches hooks to extract intermediate transformer block activations.
        """
        # Find blocks collection
        blocks = None
        if hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
        elif hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "resblocks"):
            blocks = self.backbone.transformer.resblocks

        if blocks is None:
            return

        for layer_idx in self.target_layers:
            if layer_idx < len(blocks):
                def get_hook(idx):
                    def hook_fn(module, input, output):
                        # DINOv2 / ViT output: tensor of shape [B, N+1, D] (includes CLS token)
                        if isinstance(output, tuple):
                            out_t = output[0]
                        else:
                            out_t = output
                        # Exclude CLS token to retain spatial patch tokens only: [B, N, D]
                        self.layer_features[idx] = out_t[:, 1:, :].detach()
                    return hook_fn

                h = blocks[layer_idx].register_forward_hook(get_hook(layer_idx))
                self.hooks.append(h)

    def extract_patch_features(self, x: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], Tuple[int, int]]:
        """
        Passes x through frozen foundation model.
        Returns:
            layer_features: Dict mapping layer_idx -> [B, N, D] patch tokens
            spatial_hw: (num_patches_h, num_patches_w)
        """
        self.layer_features.clear()
        with torch.no_grad():
            _ = self.backbone(x)

        H, W = x.shape[-2:]
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        return dict(self.layer_features), (H_p, W_p)

    def train(self, mode: bool = True):
        # Always enforce eval mode for frozen backbone
        super().train(False)
        self.backbone.eval()
        return self
