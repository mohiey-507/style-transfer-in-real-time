import torch
import torch.nn as nn
import torchvision.models as models
from utils import calc_mean_std

class VGGEncoder(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGEncoder, self).__init__()
        vgg19_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        # Extract up to relu4_1 (index 20)
        self.slice = nn.Sequential(*[vgg19_features[i] for i in range(21)])

        # Pre-register normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Normalize using ImageNet stats
        x = (x - self.mean) / self.std
        return self.slice(x)

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

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor):
        """
        Applies AdaIN.
        Args:
            content_feat: Features from the content image (e.g., VGG relu4_1).
            style_feat: Features from the style image (e.g., VGG relu4_1).
        Returns:
            Stylized features.
        """
        content_mean, content_std = calc_mean_std(content_feat, self.eps)
        style_mean, style_std = calc_mean_std(style_feat, self.eps)

        # Normalize content features: (x - mu_c) / sigma_c
        normalized_feat = (content_feat - content_mean) / content_std
        # Stylize: Apply style statistics: (normalized * sigma_s) + mu_s
        stylized_feat = normalized_feat * style_std + style_mean
        return stylized_feat
    
class StyleTransferNet(nn.Module):
    """
    Style Transfer Network
        Encoder -> AdaIN -> Decoder
    """
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        self.encoder = VGGEncoder(requires_grad=False)
        self.adain = AdaIN()
        self.decoder = Decoder()

    def get_adain_output(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        adain_output = self.adain(content_feat, style_feat)
        return adain_output

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        adain_target = self.get_adain_output(content, style)
        generated_img = self.decoder(adain_target)
        return generated_img, adain_target

class VGGLossExtractor(nn.Module):
    """
    Extracts features from multiple VGG19 layers (relu1_1 to relu4_1)
    for calculating style and content losses. 
        Weights are frozen.
        Input images are normalized.
    """
    def __init__(self):
        super(VGGLossExtractor, self).__init__()
        vgg19_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        self.layer_indices = {'relu1_1': 1, 'relu2_1': 6, 'relu3_1': 11, 'relu4_1': 20}
        self.max_idx = max(self.layer_indices.values())
        self.vgg_layers = nn.Sequential(*[vgg19_features[i] for i in range(self.max_idx + 1)])

        for param in self.parameters():
            param.requires_grad = False

        # Pre-register normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Normalize the input image
        x = (x - self.mean) / self.std
        features = {}
        current_feat = x
        for i, layer in enumerate(self.vgg_layers):
            current_feat = layer(current_feat)
            if i in self.layer_indices.values():
                # Name corresponding to the index
                layer_name = [name for name, index in self.layer_indices.items() if index == i][0]
                features[layer_name] = current_feat
        return features