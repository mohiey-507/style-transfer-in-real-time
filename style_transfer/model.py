import torch
import torch.nn as nn
import torchvision.models as models

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