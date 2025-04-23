import torch
import random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import config

class StyleTransferDataset(Dataset):
    def __init__(self, content_dir: str, style_dir: str, transform=None, verbose=True):
        super().__init__()
        self.transform = transform

        self.content_dir = Path(content_dir)
        self.content_files = list(self.content_dir.glob("*.jpg"))
        if verbose:
            print(f"Found {len(self.content_files)} content images in {content_dir}")

        self.style_dir = Path(style_dir)
        self.style_files = list(self.style_dir.glob("*.jpg"))
        if verbose:
            print(f"Found {len(self.style_files)} style images in {style_dir}")

    def __len__(self):
        return len(self.content_files)

    def _load_image(self, path: Path):
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading image {path}: {e}.")
            return None

    def __getitem__(self, idx):
        for offset in range(len(self)):
            c_idx = (idx + offset) % len(self)
            content = self._load_image(self.content_files[c_idx])
            style   = self._load_image(random.choice(self.style_files))
            if content and style:
                break

        if self.transform:
            content = self.transform(content)
            style   = self.transform(style)

        return content, style

def get_transform(
    image_size=config.IMAGE_SIZE,
    crop_size=config.CROP_SIZE,
    train: bool = True
):
    transforms_list = [
        v2.Resize(image_size),
    ]
    if train:
        transforms_list += [
            v2.RandomCrop(crop_size),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            v2.TrivialAugmentWide(num_magnitude_bins=31),
        ]
    else:
        transforms_list.append(v2.CenterCrop(crop_size))

    transforms_list.append(v2.ToImage())
    transforms_list.append(v2.ToDtype(torch.float32))

    return v2.Compose(transforms_list)

def get_dataloaders(
    train_content_dir:str = config.TRAIN_CONTENT_DIR,
    train_style_dir:str = config.TRAIN_STYLE_DIR,
    test_content_dir:str = config.TEST_CONTENT_DIR,
    test_style_dir:str = config.TEST_STYLE_DIR,
    batch_size:int = config.BATCH_SIZE,
    num_workers:int = config.NUM_WORKERS,
    pin_memory:bool = config.PIN_MEMORY,
    drop_last:bool = config.DROP_LAST,
) -> tuple[DataLoader, DataLoader]:
    
    train_set = StyleTransferDataset(
        content_dir=train_content_dir,
        style_dir=train_style_dir,
        transform=get_transform(train=True),
        verbose=True,
    )
    test_set = StyleTransferDataset(
        content_dir=test_content_dir,
        style_dir=test_style_dir,
        transform=get_transform(train=False),
        verbose=True,
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