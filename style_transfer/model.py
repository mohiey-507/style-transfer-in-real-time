import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Union, List

class VGGEncoder(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True):
        super(VGGEncoder, self).__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        vgg = models.vgg19(weights=weights)
        self.vgg_layers = nn.Sequential(*list(vgg.features)[:21])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.vgg_layers(x)

class VGGFeatureExtractor(VGGEncoder):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True):
        super().__init__(requires_grad=requires_grad, pretrained=pretrained)
        self.layer_indices = {
            'relu1_1': 1,
            'relu2_1': 6,
            'relu3_1': 11,
            'relu4_1': 20
        }
        self.index_to_name = {v: k for k, v in self.layer_indices.items()}

    def forward(self, x):
        x = (x - self.mean) / self.std
        features = {}

        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.index_to_name:
                features[self.index_to_name[i]] = x

        return features

class Decoder(nn.Module):
    """Mirrors the VGG encoder structure for decoding."""
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # Block 1: Upsample -> Conv -> ReLU (from 512 channels)
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, kernel_size=3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Block 2: Conv layers -> Upsample (from 256 channels)
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, kernel_size=3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, kernel_size=3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, kernel_size=3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, kernel_size=3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Block 3: Conv layers -> Upsample (from 128 channels)
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, kernel_size=3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Block 4: Conv layers -> Final Conv (from 64 channels to 3 channels)
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, kernel_size=3)
        )

    def forward(self, x):
        return self.decoder(x)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization Layer"""
    def __init__(self, eps: float = 1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def calc_mean_std(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N, C = feat.size()[:2] # N = batch size, C = channels
        feat_var = feat.view(N, C, -1).var(dim=2, unbiased=False) + self.eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor):
        """
        Applies AdaIN.
        Args:
            content_feat: Features from the content image (e.g., VGG relu4_1).
            style_feat: Features from the style image (e.g., VGG relu4_1).
        Returns:
            Stylized features.
        """
        mu_c, sigma_c = self.calc_mean_std(content_feat)
        mu_s, sigma_s = self.calc_mean_std(style_feat)

        normalized_feat = (content_feat - mu_c) / sigma_c # Normalize content features
        stylized_feat = normalized_feat * sigma_s + mu_s # Apply style statistics
        return stylized_feat

class AdaIN_Decoder(nn.Module):
    """Simple class cobining AdaIN and Decoder for training."""
    def __init__(self):
        super(AdaIN_Decoder, self).__init__()
        self.adain = AdaIN()
        self.decoder = Decoder()

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adain_target = self.adain(content_feat, style_feat)
        generated_img = self.decoder(adain_target)
        return generated_img, adain_target

class StyleTransferModel(nn.Module):
    def __init__(self, decoder_path: Union[str, Path] = None, encoder_path: Union[str, Path] = None):
        super().__init__()
        self.encoder = VGGEncoder(pretrained=False)
        self.adain = AdaIN()
        self.decoder = Decoder()

        self.encoder = self._load_with_fallback("encoder", encoder_path, self.encoder, pretrained_fallback=lambda: VGGEncoder(pretrained=True))
        self.decoder = self._load_with_fallback("decoder", decoder_path, self.decoder)

    def _load_from_path(self, path: Union[str, Path], model: nn.Module) -> nn.Module:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        sd = torch.load(path, map_location="cpu")
        if not isinstance(sd, dict):
            raise TypeError(f"Unsupported object type: {type(sd)}")
        model.load_state_dict(sd, strict=False)
        return model

    def _load_with_fallback(self, name: str, primary_path: Union[str, Path, None], model: nn.Module, pretrained_fallback=None) -> nn.Module:
        if primary_path:
            try:
                return self._load_from_path(primary_path, model)
            except Exception as e:
                print(f"Warning: failed to load from {primary_path!s} ({e}).")
        fallback = Path.cwd() / "models" / f"{name}.pth"
        try:
            if fallback.exists():
                return self._load_from_path(fallback, model)
        except Exception as e:
            print(f"Warning: failed to load from fallback {fallback!s} ({e}).")
        if pretrained_fallback is not None:
            try:
                print("Loading torch pretrained fallback (may take a while).")
                return pretrained_fallback()
            except Exception as e:
                print(f"Warning: pretrained fallback failed ({e}).")
        return model

    def forward(
        self,
        content_img: torch.Tensor,
        style_imgs: List[torch.Tensor],
        style_weights: List[float] = None,
        alpha: float = 1.0
    ) -> torch.Tensor:
        f_c = self.encoder(content_img)
        
        if len(style_imgs) == 1:
            f_s = self.encoder(style_imgs[0])
        else:
            if style_weights is None:
                style_weights = [1.0 / len(style_imgs)] * len(style_imgs)

            style_batch = torch.cat(style_imgs, dim=0)  # (N, C, H, W)
            f_s_batch = self.encoder(style_batch)

            w = torch.tensor(style_weights, device=f_s_batch.device).view(-1, 1, 1, 1)
            f_s = (f_s_batch * w).sum(dim=0, keepdim=True)  # (1, C, H, W)

        t = (1.0 - alpha) * f_c + alpha * self.adain(f_c, f_s)
        return self.decoder(t).clamp(0.0, 1.0)
