# Neural-Atari Regen

Minimal fork of Paras Chopra's Neural-Atari for rapid prototyping.  
Trains & samples Pong world model in **<3 minutes** on Colab T4.

## Quick Start

### Installation
```bash
uv pip install -r requirements.txt
```

### One-liner (Colab)
```bash
!git clone https://github.com/user/neural-atari-regen && cd neural-atari-regen && uv pip install -q -r requirements.txt && jupyter nbconvert --to notebook --execute train.ipynb
```

### Local Training
```bash
python world_model.py  # Full pipeline
```

### Jupyter Notebook
```bash
jupyter notebook train.ipynb
```

## Architecture

- **Encoder**: CNN (84×84×1 → 128D latent)
- **Decoder**: CNN (128D → 84×84×1)  
- **Predictor**: MLP (128D → 128D)
- **Training**: 500 steps AE + 500 steps predictor
- **Dataset**: d4rl-atari Pong (5K frames)

## Outputs

- `assets/real_vs_recon.png` - Reconstruction quality
- `assets/pong_fake.gif` - 200-frame rollout

## Q&A

**Why this project?**  
Rapid prototyping beats perfect code. This strips Neural-Atari to its essence: can we learn Pong dynamics in 3 minutes? Yes. The autoencoder learns paddle/ball structure, and the predictor captures basic physics. Perfect for testing ideas before scaling up.

**Future work:**  
- Action conditioning (currently pure dynamics)
- Longer horizons (>200 frames)  
- Multi-game generalization
- Hierarchical latents for complex scenes
- Differentiable physics integration

**Key learnings:**  
- Grayscale + small resolution (84×84) is sufficient for Pong
- 128D latent captures essential game state
- Simple MSE loss works well for both reconstruction and prediction
- 500 training steps each is the sweet spot for speed/quality tradeoff
- d4rl-atari provides clean, pre-processed data

**Surprises:**  
- Predictor converges faster than expected (physics is simple)
- No need for VAE regularization - deterministic AE works fine
- GIF compression makes rollouts look more stable than they are
- Single frame conditioning generates reasonable short sequences

**Next papers to read:**  
- DreamerV3: Scaling model-based RL with hierarchical world models
- GameGAN: Learning to generate interactive environments  
- VideoGPT: Video generation using VQ-VAE and transformers
- MuZero: Planning with learned models in stochastic environments
- PlaNet: A deep planning network for reinforcement learning

## File Structure
```
neural-atari-regen/
├─ world_model.py        # Core models + training loops
├─ data/d4rl_frames.py   # Dataset wrapper  
├─ train.ipynb           # Interactive training notebook
├─ assets/               # Generated outputs
├─ README.md             # This file
└─ requirements.txt      # Dependencies
```

## Performance

- **Training time**: <3 min on Colab T4
- **Reconstruction loss**: <0.08 (target)
- **Rollout length**: 200 frames
- **Memory usage**: ~2GB GPU

---

*Built for rapid iteration. Scale thoughtfully.* 