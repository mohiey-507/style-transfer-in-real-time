import config 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import calc_mean_std

class StyleTransferLoss(nn.Module):
    def __init__(self, lambda_style: float = config.LAMBDA_STYLE):
        super().__init__()
        self.lambda_style = lambda_style

        vgg19_features = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.layer_indices = {'relu1_1': 1, 'relu2_1': 6, 'relu3_1': 11, 'relu4_1': 20}
        self.max_idx = max(self.layer_indices.values())
        self.vgg_layers = nn.Sequential(*[vgg19_features[i] for i in range(self.max_idx + 1)])

        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def _extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        features = {}
        current_feat = x
        for i, layer in enumerate(self.vgg_layers):
            current_feat = layer(current_feat)
            if i in self.layer_indices.values():
                layer_name = [name for name, index in self.layer_indices.items() if index == i][0]
                features[layer_name] = current_feat
        return features

    def _calc_content_loss(self,
                        gen_features_relu4_1: dict[str, torch.Tensor],
                        adain_target_detached: dict[str, torch.Tensor]
        ) -> torch.Tensor:
        content_loss = F.mse_loss(gen_features_relu4_1, adain_target_detached)
        return content_loss

    def _calc_style_loss(self, 
                        gen_style_features: dict[str, torch.Tensor],
                        target_style_features: dict[str, torch.Tensor]
        ) -> torch.Tensor:
        style_loss = 0.0
        num_layers = len(gen_style_features)
        for layer in gen_style_features.keys():
            gen_feat = gen_style_features[layer]
            target_feat = target_style_features[layer]
            gen_mean, gen_std = calc_mean_std(gen_feat)
            target_mean, target_std = calc_mean_std(target_feat)
            style_loss += F.mse_loss(gen_mean, target_mean)
            style_loss += F.mse_loss(gen_std, target_std)
        return style_loss / num_layers 

    def forward(self,
                generated_img: torch.Tensor,
                style_img: torch.Tensor,
                adain_target: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        gen_features = self._extract_features(generated_img)
        with torch.no_grad():
            target_style_features = self._extract_features(style_img)

        content_loss = self._calc_content_loss(gen_features['relu4_1'], adain_target.detach())

        style_loss = self._calc_style_loss(gen_features, target_style_features)

        total_loss = content_loss + self.lambda_style * style_loss

        return total_loss, content_loss, style_loss