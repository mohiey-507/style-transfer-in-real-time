import torch
import random
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

class StyleTransferDataset(Dataset):
    def __init__(self, content_dir: str, style_dir: str, transform: v2.Compose, verbose=True):
        super().__init__()
        self.transform = transform

        self.content_dir = Path(content_dir)
        self.content_files = self._filter_valid_images(self.content_dir.glob("*.jpg"))
        if verbose:
            print(f"Found {len(self.content_files)} valid content images in {content_dir}")

        self.style_dir = Path(style_dir)
        self.style_files = self._filter_valid_images(self.style_dir.glob("*.jpg"))
        if verbose:
            print(f"Found {len(self.style_files)} valid style images in {style_dir}")

    def _filter_valid_images(self, file_iter):
        valid_files = []
        for path in file_iter:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_files.append(path)
            except Exception as e:
                print(f"Skipping corrupted/invalid image {path}: {e}")
        return valid_files

    def __len__(self):
        return len(self.content_files)

    def _load_image(self, path: Path):
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading image {path}: {e}")
            return None

    def __getitem__(self, idx):
        content = None
        style = None
        while (content is None or style is None):
            content = self._load_image(self.content_files[idx])
            style   = self._load_image(random.choice(self.style_files))

        content = self.transform(content)
        style   = self.transform(style)

        return content, style

def get_transform(image_size: int, crop_size: int, train: bool = True) -> v2.Compose:
    if train:
        transforms_list = [
            v2.Resize(image_size),
            v2.CenterCrop(crop_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    else:
        transforms_list = [
            v2.Resize((crop_size, crop_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    return v2.Compose(transforms_list)

def get_dataloaders(
    config: dict,
    num_workers: int = 2,
    pin_memory: bool = False,
    drop_last: bool = False,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader]:
    
    batch_size = config["BATCH_SIZE"]
    image_size = config["IMAGE_SIZE"]
    crop_size = config["CROP_SIZE"]

    train_set = StyleTransferDataset(
        content_dir=config["TRAIN_CONTENT_DIR"],
        style_dir=config["TRAIN_STYLE_DIR"],
        transform=get_transform(image_size, crop_size, train=True),
        verbose=verbose,
    )
    test_set = StyleTransferDataset(
        content_dir=config["TEST_CONTENT_DIR"],
        style_dir=config["TEST_STYLE_DIR"],
        transform=get_transform(image_size, crop_size, train=False),
        verbose=verbose,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return train_loader, test_loader
