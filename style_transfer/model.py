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