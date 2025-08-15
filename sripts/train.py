# %% [code]
!git clone https://github.com/mohamed-mohiey/style-transfer-in-real-time.git
%cd style-transfer-in-real-time

# %% [code]
!pip install -r requirements.txt

# %% [code]
import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path.cwd()))

from style_transfer.config import get_config
from style_transfer.data_setup import get_dataloaders
from style_transfer.model import AdaIN_Decoder, VGGFeatureExtractor
from style_transfer.loss import StyleTransferLoss
from style_transfer.engine import train
from style_transfer.utils import *

import torch
import torch.optim as optim

# %% [code]
set_seed(42)
config = get_config()
device = torch.device('cuda')

train_loader, val_loader = get_dataloaders(
    config, num_workers=4, pin_memory=True, drop_last=True, verbose=True
)

vgg_extractor = VGGFeatureExtractor().to(device)
decoder = AdaIN_Decoder().to(device)

loss_fn = StyleTransferLoss(lambda_style=config["LAMBDA_STYLE"]).to(device)

optimizer = optim.AdamW(decoder.parameters(), lr=config["LEARNING_RATE"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1
)

# %% [code]
history = train(
    model=decoder,
    vgg_extractor=vgg_extractor,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=config["EPOCHS"],
    save_path=config["SAVE_PATH"]
)