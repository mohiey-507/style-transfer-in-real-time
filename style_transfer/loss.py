import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import calc_mean_std

class StyleTransferLoss(nn.Module):
    def __init__(self, lambda_style: float = 12):
        super().__init__()
        self.lambda_style = lambda_style

    def _calc_style_loss(self, gen_style_features: dict[str, torch.Tensor],
                        target_style_features: dict[str, torch.Tensor]) -> torch.Tensor:

        # Initialize tensor on the same device as the features
        device = next(iter(gen_style_features.values())).device
        style_loss = torch.tensor(0.0, device=device)

        for layer in gen_style_features.keys():
            gen_feat = gen_style_features[layer]
            target_feat = target_style_features[layer]
            gen_mean, gen_std = calc_mean_std(gen_feat)
            target_mean, target_std = calc_mean_std(target_feat)
            style_loss += F.mse_loss(gen_mean, target_mean)
            style_loss += F.mse_loss(gen_std, target_std)
        return style_loss / len(gen_style_features)

    def forward(self,
            gen_features: dict[str, torch.Tensor],
            target_style_features: dict[str, torch.Tensor],
            adain_target: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        content_loss = F.mse_loss(gen_features['relu4_1'], adain_target.detach())
        style_loss = self._calc_style_loss(gen_features, target_style_features)
        total_loss = content_loss + self.lambda_style * style_loss

        return total_loss, content_loss, style_loss
