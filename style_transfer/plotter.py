import config
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image

class Plotter:
    """
    A helper class to perform style transfer inference and plot results.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = 'cpu',
        crop_size: int = config.CROP_SIZE
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.transform = self._get_inference_transform(crop_size)

    @staticmethod
    def _get_inference_transform(crop_size: int):
        """
        Inference transform: resize only to crop_size, convert to tensor and scale.
        """
        transforms_list = [
            v2.Resize(crop_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        return v2.Compose(transforms_list)

    def one_content_many_style(
        self,
        content_img: Image.Image,
        style_imgs: list[Image.Image]
    ) -> list[torch.Tensor]:
        """
        Apply one content image to multiple style images and plot results.
        """
        content_t = self.transform(content_img).unsqueeze(0).to(self.device)
        outputs = []
        with torch.inference_mode():
            for style_img in style_imgs:
                style_t = self.transform(style_img).unsqueeze(0).to(self.device)
                gen, _ = self.model(content_t, style_t)
                outputs.append(gen.squeeze(0).cpu())

        n = len(style_imgs)
        fig, axes = plt.subplots(2, n+1, figsize=(3*(n+1), 6))
        axes[0,0].axis('off')
        for i, style_img in enumerate(style_imgs, start=1):
            axes[0,i].imshow(style_img)
            axes[0,i].set_title(f"Style {i}")
            axes[0,i].axis('off')

        axes[1,0].imshow(content_img)
        axes[1,0].set_title("Content")
        axes[1,0].axis('off')
        for i, gen in enumerate(outputs, start=1):
            img = to_pil_image(gen.clamp(0,1))
            axes[1,i].imshow(img)
            axes[1,i].set_title(f"Output {i}")
            axes[1,i].axis('off')

        plt.tight_layout()
        plt.show()
        return outputs

    def many_content_one_style(
        self,
        content_imgs: list[Image.Image],
        style_img: Image.Image
    ) -> list[torch.Tensor]:
        """
        Apply multiple content images to one style image and plot results.
        """
        style_t = self.transform(style_img).unsqueeze(0).to(self.device)
        outputs = []
        with torch.inference_mode():
            for content_img in content_imgs:
                content_t = self.transform(content_img).unsqueeze(0).to(self.device)
                gen, _ = self.model(content_t, style_t)
                outputs.append(gen.squeeze(0).cpu())

        m = len(content_imgs)
        fig, axes = plt.subplots(2, m+1, figsize=(3*(m+1), 6))
        axes[0,0].axis('off')
        for i, content_img in enumerate(content_imgs, start=1):
            axes[0,i].imshow(content_img)
            axes[0,i].set_title(f"Content {i}")
            axes[0,i].axis('off')

        axes[1,0].imshow(style_img)
        axes[1,0].set_title("Style")
        axes[1,0].axis('off')
        for i, gen in enumerate(outputs, start=1):
            img = to_pil_image(gen.clamp(0,1))
            axes[1,i].imshow(img)
            axes[1,i].set_title(f"Output {i}")
            axes[1,i].axis('off')

        plt.tight_layout()
        plt.show()
        return outputs

    def many_content_many_style(
        self,
        content_imgs: list[Image.Image],
        style_imgs: list[Image.Image]
    ) -> list[tuple[Image.Image, Image.Image, torch.Tensor]]:
        """
        Apply multiple content images to multiple style images and plot results in a grid.
        """
        results = []
        with torch.inference_mode():
            for content_img in content_imgs:
                for style_img in style_imgs:
                    content_t = self.transform(content_img).unsqueeze(0).to(self.device)
                    style_t = self.transform(style_img).unsqueeze(0).to(self.device)
                    gen, _ = self.model(content_t, style_t)
                    results.append((content_img, style_img, gen.squeeze(0).cpu()))

        r, c = len(content_imgs), len(style_imgs)
        fig, axes = plt.subplots(r+1, c+1, figsize=(3*(c+1), 3*(r+1)))
        axes[0,0].axis('off')
        for j, style_img in enumerate(style_imgs, start=1):
            axes[0,j].imshow(style_img)
            axes[0,j].set_title(f"Style {j}")
            axes[0,j].axis('off')

        for i, content_img in enumerate(content_imgs, start=1):
            axes[i,0].imshow(content_img)
            axes[i,0].set_title(f"Content {i}")
            axes[i,0].axis('off')
            for j, _ in enumerate(style_imgs, start=1):
                _, _, gen = results[(i-1)*c + (j-1)]
                img = to_pil_image(gen.clamp(0,1))
                axes[i,j].imshow(img)
                axes[i,j].axis('off')

        plt.tight_layout()
        plt.show()
        return results