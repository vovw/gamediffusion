import os
import math
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
import wandb

from latent_action_data import AtariFramePairDataset
from latent_action_model import LatentActionVQVAE

# ----------------------
# Utility Functions
# ----------------------
def psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def ssim(img1, img2):
    # Simple SSIM for 3-channel images, batchwise
    # For full SSIM, use torchmetrics or skimage
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = img1.mean(dim=[2,3], keepdim=True)
    mu2 = img2.mean(dim=[2,3], keepdim=True)
    sigma1 = ((img1 - mu1) ** 2).mean(dim=[2,3], keepdim=True)
    sigma2 = ((img2 - mu2) ** 2).mean(dim=[2,3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2,3], keepdim=True)
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean().item()

def codebook_entropy(indices, codebook_size):
    # indices: (B, H, W) or (N,)
    hist = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    prob = hist / hist.sum()
    entropy = -(prob[prob > 0] * prob[prob > 0].log()).sum().item()
    return entropy, hist.cpu().numpy()

def save_checkpoint(state, is_best, checkpoint_dir, step):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_{step:06d}.pt")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best.pt"))

# ----------------------
# Training Function
# ----------------------
def train(args):
       # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # CUDA-specific optimization settings
        compile_mode = "default"  # Use the default backend for CUDA
        use_amp = True  # Use automatic mixed precision for CUDA
        pin_memory = True  # Use pinned memory for faster data transfer
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        # MPS-specific settings
        compile_mode = "eager"  # Use eager mode for MPS compatibility
        use_amp = False  # MPS may have issues with automatic mixed precision
        pin_memory = False  # pin_memory not supported on MPS
    else:
        device = torch.device('cpu')
        # CPU settings
        compile_mode = None  # Skip compilation on CPU for simplicity
        use_amp = False  # No need for mixed precision on CPU
        pin_memory = False  # No need for pinned memory on CPU
    print(f"Using device: {device}")

    # WandB setup
    wandb.init(project="latent-action", config=vars(args))

    # Data
    train_set = AtariFramePairDataset(args.data_dir, split='train', grayscale=args.grayscale)
    val_set = AtariFramePairDataset(args.data_dir, split='val', grayscale=args.grayscale)
    test_set = AtariFramePairDataset(args.data_dir, split='test', grayscale=args.grayscale)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    # Print data statistics
    def get_stats(dataset, name):
        n_episodes = len(dataset.episode_dirs)
        n_pairs = len(dataset)
        sample = dataset[0]
        shape = sample[0].shape
        print(f"{name}: {n_episodes} episodes, {n_pairs} pairs, frame shape: {shape}")
    print("--- Data Statistics ---")
    get_stats(train_set, "Train")
    get_stats(val_set, "Val")
    get_stats(test_set, "Test")
    print("-----------------------")

    # Model
    model = LatentActionVQVAE()
    model = model.to(device)
    # Conditionally compile the model
    if compile_mode is not None:
        try:
            print(f"Compiling model with {compile_mode} backend...")
            model = torch.compile(model, backend=compile_mode)
            print("Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing without compilation")

    # Optimizer & Scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iters, eta_min=args.lr_min)

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    # Training state
    best_val_loss = float('inf')
    global_step = 0
    grad_accum = args.grad_accum
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'latent_action')

    model.train()
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    pbar = tqdm(range(args.max_iters), desc="Training", dynamic_ncols=True)
    while global_step < args.max_iters:
        try:
            frame_t, frame_tp1 = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            frame_t, frame_tp1 = next(data_iter)
        frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)
        with torch.amp.autocast(device.type, enabled=(scaler is not None)):
            recon, indices, commit_loss, codebook_loss = model(frame_t, frame_tp1)
            rec_loss = F.mse_loss(recon, frame_tp1)
            # Entropy regularization
            entropy, hist = codebook_entropy(indices, 256)
            entropy_reg = -args.entropy_weight * entropy
            loss = rec_loss + commit_loss + codebook_loss + entropy_reg
        if scaler is not None:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()
        if (global_step + 1) % grad_accum == 0:
            if args.grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        # Logging
        if (global_step + 1) % args.log_interval == 0:
            with torch.no_grad():
                psnr_val = psnr(recon, frame_tp1)
                ssim_val = ssim(recon, frame_tp1)
                l1 = F.l1_loss(recon, frame_tp1).item()
                l2 = F.mse_loss(recon, frame_tp1).item()
                wandb.log({
                    'loss/total': loss.item(),
                    'loss/rec': rec_loss.item(),
                    'loss/commit': commit_loss.item(),
                    'loss/codebook': codebook_loss.item(),
                    'loss/entropy_reg': entropy_reg,
                    'metric/psnr': psnr_val,
                    'metric/ssim': ssim_val,
                    'metric/l1': l1,
                    'metric/l2': l2,
                    'codebook/entropy': entropy,
                    'codebook/hist': wandb.Histogram(hist),
                    'lr': scheduler.get_last_lr()[0],
                    'step': global_step + 1
                }, step=global_step + 1)
        # Save reconstructions
        if (global_step + 1) % args.recon_interval == 0:
            with torch.no_grad():
                grid = make_grid(torch.cat([frame_t, frame_tp1, recon], dim=0), nrow=args.batch_size)
                wandb.log({'reconstructions': [wandb.Image(grid, caption=f'Step {global_step+1}')]}, step=global_step+1)
        # Codebook reset
        if (global_step + 1) % args.codebook_reset_interval == 0:
            with torch.no_grad():
                used = (hist > 0).sum()
                if used < 0.5 * 256:
                    print(f"Resetting unused codebook entries at step {global_step+1}")
                    model.vq.embeddings.weight.data[hist == 0] = torch.randn_like(model.vq.embeddings.weight.data[hist == 0])
        # Checkpoint
        if (global_step + 1) % args.ckpt_interval == 0:
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': global_step + 1
            }, is_best=False, checkpoint_dir=checkpoint_dir, step=global_step + 1)
        # Validation
        if (global_step + 1) % args.val_interval == 0:
            val_loss = evaluate(model, val_loader, device, scaler, pbar_desc='Val')
            wandb.log({'val/loss': val_loss}, step=global_step+1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': global_step + 1
                }, is_best=True, checkpoint_dir=checkpoint_dir, step=global_step + 1)
        global_step += 1
        pbar.update(1)
    pbar.close()
    # Final test evaluation
    test_loss = evaluate(model, test_loader, device, scaler, pbar_desc='Test')
    wandb.log({'test/loss': test_loss}, step=global_step)
    print(f"Test loss: {test_loss}")

def evaluate(model, loader, device, scaler, pbar_desc=None):
    model.eval()
    losses = []
    with torch.no_grad():
        if pbar_desc:
            loader_iter = tqdm(loader, desc=pbar_desc, dynamic_ncols=True)
        else:
            loader_iter = loader
        for frame_t, frame_tp1 in loader_iter:
            frame_t, frame_tp1 = frame_t.to(device), frame_tp1.to(device)
            with torch.amp.autocast(device.type, enabled=(scaler is not None)):
                recon, indices, commit_loss, codebook_loss = model(frame_t, frame_tp1)
                rec_loss = F.mse_loss(recon, frame_tp1)
                loss = rec_loss + commit_loss + codebook_loss
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='bulk_videos', help='Directory with episode folders')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grayscale', action='store_true', default=False)
    parser.add_argument('--entropy_weight', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--recon_interval', type=int, default=2500)
    parser.add_argument('--ckpt_interval', type=int, default=10000)
    parser.add_argument('--val_interval', type=int, default=1000)
    parser.add_argument('--codebook_reset_interval', type=int, default=10000)
    args = parser.parse_args()
    train(args) 