import torch
import numpy as np
from PIL import Image
import imageio
import random
from latent_action_model import load_latent_action_model, ActionToLatentMLP

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

action_to_latent = ActionToLatentMLP().to(device)
ckpt = torch.load('checkpoints/latent_action/action_to_latent_best.pt', map_location=device)
# Fix state dict keys by removing the '_orig_mod.' prefix if present
fixed_state_dict = {}
for k, v in ckpt['model_state_dict'].items():
    if k.startswith('_orig_mod.'):
        fixed_state_dict[k.replace('_orig_mod.', '')] = v
    else:
        fixed_state_dict[k] = v
action_to_latent.load_state_dict(fixed_state_dict)
action_to_latent.eval()
if device.type == 'cuda':
    action_to_latent = torch.compile(action_to_latent)
print(f"[DEBUG] Loaded action-to-latent model.")

# --- Action mapping ---
action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
def action_to_onehot(action_idx):
    onehot = torch.zeros(1, 4, device=device)
    onehot[0, action_idx] = 1.0
    return onehot

def map_action_to_latent_embedding(action_idx, temperature=1.0):
    onehot = action_to_onehot(action_idx)
    print(f"[DEBUG] Action idx: {action_idx}, onehot: {onehot.cpu().numpy()}")
    with torch.no_grad():
        logits = action_to_latent(onehot)  # (1, 35, 256)
    print(f"[DEBUG] Logits shape: {logits.shape}, dtype: {logits.dtype}")
    with torch.no_grad():
        indices = action_to_latent.sample_latents(logits, temperature=temperature)  # (1, 35)
    print(f"[DEBUG] Sampled latent indices shape: {indices.shape}, min: {indices.min().item()}, max: {indices.max().item()}")
    indices = indices.view(1, 5, 7)  # (1, 5, 7)
    print(f"[DEBUG] Reshaped latent indices: {indices.shape}")
    # Get quantized embedding from codebook
    embeddings = world_model.vq.embeddings  # nn.Embedding(256, 128)
    print(f"[DEBUG] indices device: {indices.device}, embeddings device: {embeddings.weight.device}")
    indices = indices.to(embeddings.weight.device)
    quantized = embeddings(indices)  # (1, 5, 7, 128)
    print(f"[DEBUG] Quantized embedding shape (before permute): {quantized.shape}")
    quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (1, 128, 5, 7)
    print(f"[DEBUG] Quantized embedding shape (after permute): {quantized.shape}")
    return quantized

# --- Frame generation loop ---
frames = []
current_frame = init_frame.clone()

current_frame_for_frame = (current_frame.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
frames.append(current_frame_for_frame)

for step in range(100):
    print(f"\n[DEBUG] === Step {step} ===")
    action_idx = random.randint(0, 3)
    print(f"[DEBUG] Random action: {action_idx} ({action_names[action_idx]})")
    latent = map_action_to_latent_embedding(action_idx)
    # Decoder expects (B, 128, 5, 7) and (B, 3, 160, 210)
    with torch.no_grad():
        # Permute current_frame to (B, 3, 160, 210)
        frame_in = current_frame.permute(0, 1, 3, 2)  # (1, 3, 210, 160) -> (1, 3, 160, 210)
        # Ensure both latent and frame_in are on the same device as world_model
        latent = latent.to(next(world_model.parameters()).device)
        frame_in = frame_in.to(next(world_model.parameters()).device)
        print(f"[DEBUG] Frame_in shape for decoder: {frame_in.shape}, device: {frame_in.device}")
        print(f"[DEBUG] Latent shape for decoder: {latent.shape}, device: {latent.device}")
        next_frame = world_model.decoder(latent, frame_in)  # (1, 3, 160, 210)
        print(f"[DEBUG] Decoder output shape: {next_frame.shape}, device: {next_frame.device}")
        # Permute back to (1, 3, 210, 160)
        next_frame = next_frame.permute(0, 1, 3, 2)
        print(f"[DEBUG] Next frame shape (after permute): {next_frame.shape}")
        # Clamp and convert to uint8 for saving
        frame_np = (next_frame.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        print(f"[DEBUG] Frame_np shape for video: {frame_np.shape}, dtype: {frame_np.dtype}")
        frames.append(frame_np)
        current_frame = next_frame.clone()

# --- Save video ---
print(f"[DEBUG] Saving video with {len(frames)} frames...")
imageio.mimsave('data/neural_random_game.gif', frames, fps=3)
print('Saved video to data/neural_random_game.gif') 