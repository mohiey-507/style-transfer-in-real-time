
import sys
import torch
import argparse
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_transfer.model import StyleTransferModel

def get_transform(crop_size: int = 256) -> v2.Compose:
    """Returns a transform to prepare images for the model."""
    return v2.Compose([
        v2.Resize((crop_size, crop_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

def load_image(image_path: str, transform: v2.Compose) -> torch.Tensor:
    """Loads an image and applies transformations."""
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Style Transfer.")
    parser.add_argument('-content', type=str, required=True, help="Path to the content image.")
    parser.add_argument('-style', type=str, required=True, nargs='+', help="Path(s) to the style image(s).")
    parser.add_argument('-interp_weight', type=float, nargs='+', help="Interpolation weights for the style images.")
    parser.add_argument('-alpha', type=float, default=1.0, help="Style interpolation factor (0.0 to 1.0).")
    parser.add_argument('-crop', type=int, default=256, help="Size to crop the images to.")
    parser.add_argument('-decoder_path', type=str, default=None, help="Path to the trained decoder model.")
    parser.add_argument('-encoder_path', type=str, default=None, help="Path to the trained encoder model.")
    parser.add_argument('-output', type=str, default='output.jpg', help="Path to save the output image.")
    
    args = parser.parse_args()

    if args.interp_weight and len(args.style) != len(args.interp_weight):
        print("Error: The number of style images must match the number of interpolation weights.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = get_transform(args.crop)
    
    content_img = load_image(args.content, transform).to(device)
    style_imgs = [load_image(s, transform).to(device) for s in args.style]
    
    model = StyleTransferModel(encoder_path=args.encoder_path, decoder_path=args.decoder_path).to(device)
    
    with torch.inference_mode():
        output = model(content_img, style_imgs, style_weights=args.interp_weight, alpha=args.alpha)
        
    save_image(output, args.output)
    print(f"Output image saved to {args.output}")

if __name__ == '__main__':
    main()
