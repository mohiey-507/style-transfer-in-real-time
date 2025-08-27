import sys
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_transfer.model import StyleTransferModel

from run import get_transform, load_image

def soft_interp_grid(n: int = 5, temperature: float = 0.5):
    u = torch.linspace(0, 1, n)
    v = torch.linspace(0, 1, n)
    U, V = torch.meshgrid(u, v, indexing="xy")  # [n, n]

    corners = torch.tensor([
        [0., 0.],  # TL
        [1., 0.],  # TR
        [0., 1.],  # BL
        [1., 1.],  # BR
    ])

    # Stack grid points → [n, n, 2]
    grid_points = torch.stack([U, V], dim=-1)

    # Compute squared distance to each corner → [n, n, 4]
    d2 = torch.sum((grid_points[..., None, :] - corners)**2, dim=-1)

    # Softmax over -distance/temperature
    weights = torch.softmax(-d2 / temperature, dim=-1)

    return weights

def load_with_fallback(image_path: str, transform, device, default_path: str):
    """Load image with fallback to default."""
    if image_path:
        try:
            return load_image(image_path, transform).to(device)
        except:
            print(f"Warning: Failed to load {image_path}, using default {default_path}")
    try:
        return load_image(default_path, transform).to(device)
    except Exception as e:
        print(f"Error: Failed to load default {default_path}: {e}")
        sys.exit(1)

def generate_tradeoff_figure(content_img: torch.Tensor, style_img: torch.Tensor, model: StyleTransferModel) -> plt.Figure:
    """Generate content-style trade-off figure (1x6)."""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    images = []
    for a in alphas:
        output = model(content_img, [style_img], alpha=a)
        images.append(output.squeeze(0).cpu().permute(1, 2, 0).numpy())
    images.append(style_img.squeeze(0).cpu().permute(1, 2, 0).numpy())  # Pure style

    fig, axs = plt.subplots(1, 6, figsize=(18, 3))
    fig.subplots_adjust(wspace=0.1)
    for i, (img, ax) in enumerate(zip(images, axs)):
        ax.imshow(img)
        ax.axis('off')
        if i < 5:
            ax.set_title(f"α={alphas[i]}")
        else:
            ax.set_title("Style")
    return fig

def generate_interpolation_figure(content_img: torch.Tensor, style_imgs: list[torch.Tensor], model: StyleTransferModel, alpha: float, temp: float) -> plt.Figure:
    """Generate style interpolation figure (5x7)."""
    # Styles order: [TL=0, BL=1, TR=2, BR=3]
    interp_styles = [style_imgs[0], style_imgs[2], style_imgs[1], style_imgs[3]]  # Remap to [TL, TR, BL, BR]
    weights = soft_interp_grid(n=5, temperature=temp)

    interp_images = []
    for i in range(5):
        row = []
        for j in range(5):
            w = weights[i, j].tolist()
            output = model(content_img, interp_styles, style_weights=w, alpha=alpha)
            row.append(output.squeeze(0).cpu().permute(1, 2, 0).numpy())
        interp_images.append(row)

    fig, axs = plt.subplots(5, 7, figsize=(21, 15))
    fig.subplots_adjust(wspace=0, hspace=0) 

    # Place corner styles
    axs[0, 0].imshow(style_imgs[0].squeeze(0).cpu().permute(1, 2, 0).numpy())  # TL
    axs[4, 0].imshow(style_imgs[1].squeeze(0).cpu().permute(1, 2, 0).numpy())  # BL
    axs[0, 6].imshow(style_imgs[2].squeeze(0).cpu().permute(1, 2, 0).numpy())  # TR
    axs[4, 6].imshow(style_imgs[3].squeeze(0).cpu().permute(1, 2, 0).numpy())  # BR
    for ax in [axs[0,0], axs[4,0], axs[0,6], axs[4,6]]:
        ax.axis('off')

    # Hide empty cells
    for r in [1,2,3]:
        axs[r, 0].axis('off')
        axs[r, 6].axis('off')

    # Place interp grid
    for i in range(5):
        for j in range(5):
            axs[i, j+1].imshow(interp_images[i][j])
            axs[i, j+1].axis('off')

    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate style transfer plots.")
    parser.add_argument('-content', type=str, default=None, help="Content image path (shared or fallback per figure)")
    parser.add_argument('-style_tradeoff', type=str, default=None, help="Style for trade-off")
    parser.add_argument('-styles', type=str, nargs=4, default=None, help="4 style paths for interpolation")
    parser.add_argument('-alpha', type=float, default=0.70, help="Alpha for interpolation")
    parser.add_argument('-crop', type=int, default=512, help="Crop size")
    parser.add_argument('-temp', type=float, default=0.30, help="Temperature for style_weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    transform = get_transform(args.crop)
    model = StyleTransferModel().to(device)

    # Load for trade-off
    default_content_trade = 'examples/content_style_alpha/chicago.jpg'
    default_style_trade = 'examples/content_style_alpha/ashville.jpg'
    content_trade = load_with_fallback(args.content, transform, device, default_content_trade)
    style_trade = load_with_fallback(args.style_tradeoff, transform, device, default_style_trade)

    # Load for interpolation
    default_content_interp = 'examples/style_interpolation/avril.jpg'
    default_styles = [
        'examples/style_interpolation/the_starry_night.png',  # TL
        'examples/style_interpolation/on_the_island.png',     # BL
        'examples/style_interpolation/gustav_klimt_the_kiss.png',  # TR
        'examples/style_interpolation/claude_monet_impression_sunrise.png'  # BR
    ]
    content_interp = load_with_fallback(args.content, transform, device, default_content_interp)
    if args.styles:
        try:
            styles_interp = [load_image(p, transform).to(device) for p in args.styles]
            if len(styles_interp) != 4:
                raise ValueError
        except:
            print("Warning: Invalid styles provided, using defaults")
            styles_interp = [load_image(p, transform).to(device) for p in default_styles]
    else:
        styles_interp = [load_image(p, transform).to(device) for p in default_styles]

    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    with torch.inference_mode():
        fig_trade = generate_tradeoff_figure(content_trade, style_trade, model)
        fig_trade.savefig(output_dir / 'tradeoff.png', bbox_inches='tight')
        print(f"Trade-off figure saved to {output_dir / 'tradeoff.png'}")

        fig_interp = generate_interpolation_figure(content_interp, styles_interp, model, args.alpha, args.temp)
        fig_interp.savefig(output_dir / 'interpolation.png', bbox_inches='tight')
        print(f"Interpolation figure saved to {output_dir / 'interpolation.png'}")

if __name__ == '__main__':
    main()
