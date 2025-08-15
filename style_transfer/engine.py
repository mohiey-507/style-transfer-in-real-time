import os 
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_step(
        model: nn.Module,
        vgg_extractor: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scaler: torch.GradScaler,
        device: torch.device,
        train_loader: DataLoader,
) -> tuple[float, float, float]:
    model.train()
    train_total_loss = 0.0
    train_content_loss = 0.0
    train_style_loss = 0.0

    num_batches = len(train_loader)
    train_pbar = tqdm.tqdm(train_loader, desc=f"[Train]", leave=False)

    for content, style in train_pbar:
        content, style  = content.to(device), style.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type):
            content_features = vgg_extractor(content)['relu4_1']
            style_features = vgg_extractor(style)

            generated_img, adain_target = model(content_features, style_features['relu4_1'])

            gen_features = vgg_extractor(generated_img)
            total_loss, content_loss, style_loss = loss_fn(
                gen_features, style_features, adain_target
            )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_total_loss += total_loss.item()
        train_content_loss += content_loss.item()
        train_style_loss += style_loss.item()

        train_pbar.set_postfix({
            "Total L": f"{total_loss.item():.4f}",
            "Content L": f"{content_loss.item():.4f}",
            "Style L": f"{style_loss.item():.4f}"
        })

    train_pbar.close()
    return (train_total_loss / num_batches,
            train_content_loss / num_batches,
            train_style_loss / num_batches)

def validate_step(
        model: nn.Module,
        vgg_extractor: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        val_loader: DataLoader,
) -> tuple[float, float, float]:
    model.eval()
    val_total_loss = 0.0
    val_content_loss = 0.0
    val_style_loss = 0.0

    num_batches = len(val_loader)
    val_pbar = tqdm.tqdm(val_loader, desc=f"[Val]", leave=False)

    with torch.inference_mode():
        for content, style in val_pbar:
            content, style = content.to(device), style.to(device)

            with torch.autocast(device_type=device.type):
                content_features = vgg_extractor(content)['relu4_1']
                style_features = vgg_extractor(style)

                generated_img, adain_target = model(content_features, style_features['relu4_1'])

                gen_features = vgg_extractor(generated_img)
                total_loss, content_loss, style_loss = loss_fn(
                    gen_features, style_features, adain_target
                )


            val_total_loss += total_loss.item()
            val_content_loss += content_loss.item()
            val_style_loss += style_loss.item()

            val_pbar.set_postfix({
                "Total L": f"{total_loss.item():.4f}",
                "Content L": f"{content_loss.item():.4f}",
                "Style L": f"{style_loss.item():.4f}"
            })

    val_pbar.close()
    return (val_total_loss / num_batches,
            val_content_loss / num_batches,
            val_style_loss / num_batches)

def train(
        model: nn.Module,
        vgg_extractor: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int,
        save_path: str = None,
) -> dict:
    print(f"Starting training on {device} for {epochs} epochs...")

    scaler = torch.GradScaler()

    history = {
        'train_total_loss': [],
        'train_content_loss': [],
        'train_style_loss': [],
        'val_total_loss': [],
        'val_content_loss': [],
        'val_style_loss': [],
        'lr': [],
    }
    best_val_loss = float('inf')

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):

        t_total, t_content, t_style = train_step(
            model=model, vgg_extractor=vgg_extractor, loss_fn=loss_fn, 
            optimizer=optimizer, scaler=scaler, device=device, 
            train_loader=train_loader
        )

        v_total, v_content, v_style = validate_step(
            model=model, vgg_extractor=vgg_extractor, loss_fn=loss_fn,
            device=device, val_loader=val_loader
        )
        scheduler.step(v_total)

        history['train_total_loss'].append(t_total)
        history['train_content_loss'].append(t_content)
        history['train_style_loss'].append(t_style)
        history['val_total_loss'].append(v_total)
        history['val_content_loss'].append(v_content)
        history['val_style_loss'].append(v_style)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print("-" * 60)
        print(f"Epoch {epoch+1}/{epochs} Completed:")
        print(f"  Train Loss -> Total: {t_total:.4f} | Content: {t_content:.4f} | Style: {t_style:.4f}")
        print(f"  Val Loss   -> Total: {v_total:.4f} | Content: {v_content:.4f} | Style: {v_style:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.4f}")
        print("-" * 60)

        if save_path and v_total < best_val_loss:
            best_val_loss = v_total
            full_save_path = os.path.join(save_path, f"decoder.pth")
            try:
                torch.save(model.decoder.state_dict(), full_save_path)
                print(f"  Decoder saved to {full_save_path} (Val Loss: {best_val_loss:.4f})")
            except Exception as e:
                print(f"  Error saving decoder: {e}")
        print("Training finished.")
    return history
