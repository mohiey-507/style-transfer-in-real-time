import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Union

class VGGEncoder(nn.Module):
    def __init__(self, requires_grad: bool = False):
        super(VGGEncoder, self).__init__()
        vgg19_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Extract up to relu4_1
        self.vgg_layers = nn.Sequential(*[vgg19_features[i] for i in range(21)])

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.vgg_layers(x)

class VGGFeatureExtractor(VGGEncoder):
    def __init__(self, requires_grad: bool = False):
        super().__init__(requires_grad=requires_grad)
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
    def __init__(self, decoder_path: str = None):
        super().__init__()
        self.encoder = VGGEncoder()
        self.adain = AdaIN()
        self.decoder = Decoder()

        path = Path(decoder_path) if decoder_path else Path.cwd() / "models" / "decoder.pth"
        try:
            self.load_decoder_from_path(path)
        except Exception as e1:
            print(f"Warning: failed to load decoder from {path!s} ({e1}).")
            fallback = Path.cwd() / "models" / "decoder.pth"
            if fallback != path:
                try:
                    self.load_decoder_from_path(fallback)
                except Exception as e2:
                    print(f"Failed fallback {fallback!s} ({e2}).")

    def load_decoder_from_path(self, path: Union[str, Path]):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            self.decoder.load_state_dict(obj)
        elif isinstance(obj, nn.Module):
            self.decoder = obj
        else:
            raise TypeError(f"Unsupported decoder object type: {type(obj)}")

    def forward(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        f_c = self.encoder(content_img)
        f_s = self.encoder(style_img)
        t = (1.0 - alpha) * f_c + alpha * self.adain(f_c, f_s)
        return self.decoder(t).clamp(0.0, 1.0)
