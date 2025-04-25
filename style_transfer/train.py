from config import *
from model import StyleTransferNet
from data_setup import get_dataloaders
from loss import StyleTransferLoss
from engine import train

import torch
import torch.optim as optim

train_loader, val_loader = get_dataloaders()
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
st_model = StyleTransferNet().to(device)
loss_fn = StyleTransferLoss(lambda_style=LAMBDA_STYLE).to(device)
optimizer = optim.Adam(st_model.decoder.parameters(), lr=LEARNING_RATE)

history = train(
        model=st_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )