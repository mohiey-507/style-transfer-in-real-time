import config 
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_step(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        train_pbar: tqdm.tqdm,
) -> tuple[float, float, float]:
    """Performs a single training step."""
    model.train()
    train_total_loss = 0.0
    train_content_loss = 0.0
    train_style_loss = 0.0

    for content, style in train_pbar:

        content, style  = content.to(device), style.to(device)

        optimizer.zero_grad()

        generated_img, adain_target = model(content, style)

        total_loss, content_loss, style_loss = loss_fn(
            generated_img, style, adain_target
        )

        total_loss.backward()
        optimizer.step()

        train_total_loss += total_loss.item()
        train_content_loss += content_loss.item()
        train_style_loss += style_loss.item()

        train_pbar.set_postfix({
            "Total L": f"{total_loss.item():.4f}",
            "Content L": f"{content_loss.item():.4f}",
            "Style L": f"{style_loss.item():.4f}"
        })

    num_batches = len(train_pbar)
    return (train_total_loss / num_batches,
            train_content_loss / num_batches,
            train_style_loss / num_batches)

def validate_step(
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        val_pbar: tqdm.tqdm,
) -> tuple[float, float, float]:
    """Performs a single validation step."""
    model.eval()
    val_total_loss = 0.0
    val_content_loss = 0.0
    val_style_loss = 0.0

    with torch.inference_mode():
        for content, style in val_pbar:

            content, style  = content.to(device), style.to(device)

            generated_img, adain_target = model(content, style)

            total_loss, content_loss, style_loss = loss_fn(
                generated_img, style, adain_target
            )

            val_total_loss += total_loss.item()
            val_content_loss += content_loss.item()
            val_style_loss += style_loss.item()

            val_pbar.set_postfix({
                "Total L": f"{total_loss.item():.4f}",
                "Content L": f"{content_loss.item():.4f}",
                "Style L": f"{style_loss.item():.4f}"
            })

    num_batches = len(val_pbar)
    return (val_total_loss / num_batches,
            val_content_loss / num_batches,
            val_style_loss / num_batches)

def train(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = config.DEVICE,
        epochs: int = config.EPOCHS,
        save_path: str = config.SAVE_PATH,
) -> dict:
    print(f"Starting training on {device} for {epochs} epochs...")

    history = {
        'train_total_loss': [],
        'train_content_loss': [],
        'train_style_loss': [],
        'val_total_loss': [],
        'val_content_loss': [],
        'val_style_loss': []
    }
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- Training ---
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        t_total, t_content, t_style = train_step(
            model, loss_fn, optimizer, device, train_pbar
        )
        train_pbar.close()

        # --- Validation ---
        val_pbar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        v_total, v_content, v_style = validate_step(
            model, loss_fn, device, val_pbar
        )
        val_pbar.close()

        # --- logging ---
        history['train_total_loss'].append(t_total)
        history['train_content_loss'].append(t_content)
        history['train_style_loss'].append(t_style)
        history['val_total_loss'].append(v_total)
        history['val_content_loss'].append(v_content)
        history['val_style_loss'].append(v_style)

        print("-" * 60)
        print(f"Epoch {epoch+1}/{epochs} Completed:")
        print(f"  Train Loss -> Total: {t_total:.4f} | Content: {t_content:.4f} | Style: {t_style:.4f}")
        print(f"  Val Loss   -> Total: {v_total:.4f} | Content: {v_content:.4f} | Style: {v_style:.4f}")
        print("-" * 60)

        # Save best model
        if save_path and v_total < best_val_loss:
            best_val_loss = v_total
            full_save_path = f"{save_path}_best_model.pth"
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, full_save_path)
                print(f"  Model saved to {full_save_path} (Val Loss: {best_val_loss:.4f})")
            except Exception as e:
                print(f"  Error saving model: {e}")

    print("Training finished.")
    return history