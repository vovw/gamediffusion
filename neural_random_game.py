import torch
import numpy as np
from PIL import Image
import imageio
import random
import argparse
from latent_action_model import load_latent_action_model, ActionToLatentMLP, ActionStateToLatentMLP

# --- Device selection ---
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"[DEBUG] Using device: {device}")

# --- Load initial frame ---
init_img = Image.open('data/0.png').convert('RGB')
init_frame = np.array(init_img, dtype=np.float32) / 255.0  # (210, 160, 3), [0,1]
print(f"[DEBUG] Loaded initial frame shape (np): {init_frame.shape}")
init_frame = torch.from_numpy(init_frame).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 210, 160)
print(f"[DEBUG] Initial frame tensor shape: {init_frame.shape}, dtype: {init_frame.dtype}")

# --- Load models ---
world_model, _ = load_latent_action_model('checkpoints/latent_action/best.pt', device)
world_model.to(device)
world_model.eval()
if device.type == 'cuda':
    world_model = torch.compile(world_model)
print(f"[DEBUG] Loaded world model. VQ codebook shape: {world_model.vq.embeddings.weight.shape}, device: {next(world_model.parameters()).device}")

def action_to_onehot(action_idx, device):
    onehot = torch.zeros(1, 4, device=device)
    onehot[0, action_idx] = 1.0
    return onehot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_frames', action='store_true', help='Use action+state (frames) model for action-to-latent')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to generate')
    parser.add_argument('--temperature', type=float, default=0.01, help='Sampling temperature for latent prediction')
    args = parser.parse_args()

    # --- Load action-to-latent model ---
    if args.with_frames:
        print("[INFO] Using ActionStateToLatentMLP (action + last 2 frames)")
        model = ActionStateToLatentMLP().to(device)
        ckpt = torch.load('checkpoints/latent_action/action_state_to_latent_best.pt', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("[INFO] Using ActionToLatentMLP (action only)")
        model = ActionToLatentMLP().to(device)
        ckpt = torch.load('checkpoints/latent_action/action_to_latent_best.pt', map_location=device)
        # Fix state dict keys by removing the '_orig_mod.' prefix if present
        fixed_state_dict = {}
        for k, v in ckpt['model_state_dict'].items():
            if k.startswith('_orig_mod.'):
                fixed_state_dict[k.replace('_orig_mod.', '')] = v
            else:
                fixed_state_dict[k] = v
        model.load_state_dict(fixed_state_dict)
    model.eval()
    if device.type == 'cuda':
        model = torch.compile(model)

    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    frames = []
    current_frame = init_frame.clone()
    current_frame_for_frame = (current_frame.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    frames.append(current_frame_for_frame)

    if args.with_frames:
        # Frame buffer: last 2 frames (each (1, 3, 210, 160))
        last2_frames = [current_frame.clone(), current_frame.clone()]

    for step in range(args.steps):
        print(f"\n[DEBUG] === Step {step} ===")
        action_idx = random.randint(0, 3)
        print(f"[DEBUG] Random action: {action_idx} ({action_names[action_idx]})")
        with torch.no_grad():
            if args.with_frames:
                # Stack last 2 frames: (1, 6, 210, 160)
                stacked_frames = torch.cat([last2_frames[0], last2_frames[1]], dim=1)  # (1, 6, 210, 160)
                onehot = action_to_onehot(action_idx, device)
                logits = model(onehot, stacked_frames)
                indices = model.sample_latents(logits, temperature=args.temperature)
            else:
                onehot = action_to_onehot(action_idx, device)
                logits = model(onehot)
                indices = model.sample_latents(logits, temperature=args.temperature)
            indices = indices.view(1, 5, 7)
            embeddings = world_model.vq.embeddings
            indices = indices.to(embeddings.weight.device)
            quantized = embeddings(indices)  # (1, 5, 7, 128)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (1, 128, 5, 7)
            # Frame for decoder
            frame_in = current_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160) -> (1, 3, 160, 210)
            quantized = quantized.to(next(world_model.parameters()).device)
            frame_in = frame_in.to(next(world_model.parameters()).device)
            next_frame = world_model.decoder(quantized, frame_in)  # (1, 3, 160, 210)
            next_frame = next_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160)
            frame_np = (next_frame.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            frames.append(frame_np)
            current_frame = next_frame.clone()
            if args.with_frames:
                last2_frames[0] = last2_frames[1]
                last2_frames[1] = current_frame.clone()

    print(f"[DEBUG] Saving video with {len(frames)} frames...")
    imageio.mimsave('data/neural_random_game.gif', frames, fps=3)
    print('Saved video to data/neural_random_game.gif')

if __name__ == '__main__':
    main() 