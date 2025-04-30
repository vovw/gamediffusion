# Part 2: Latent Action Prediction Model

## Overview
This part focuses on building a model that can infer latent actions between consecutive frames. Following the LAPA paper, we'll use a VQ-VAE-inspired approach to learn a discrete codebook of latent actions that explain the transitions between frames.

## Related Components
- [part1.md](part1.md): Provides the raw video data for training
- [part3.md](part3.md): Uses the latent actions from this model to predict next frames
- [part4.md](part4.md): Maps these latent actions to actual game controls

## Implementation Details

### Data Preparation
1. Load consecutive frame pairs from gameplay recordings
2. Create a PyTorch dataset of (frame_t, frame_t+1) pairs
3. Apply minimal preprocessing:
   - Normalize pixel values to [0, 1]
   - No need for data augmentation as we want to learn the actual game dynamics

### Latent Action Model Architecture
1. Implement a VQ-VAE based model with:
   - Encoder: Takes both current frame and next frame as input
     - 4 convolutional layers with stride 2 (downsampling)
     - Channel progression: 1→64→128→256→512
     - ReLU activations and batch normalization
   
   - Vector Quantization Layer:
     - Codebook size: 256 vectors (8-bit)
     - Embedding dimension: 64
     - Implement with straight-through estimator for backpropagation
     - Commitment loss weight: 0.25
   
   - Decoder: Takes current frame and quantized latent code as input
     - Cross attention mechanism to incorporate the latent action
     - 4 transposed convolutional layers (upsampling)
     - Channel progression: 512→256→128→64→1
     - Skip connections from encoder (U-Net style)

2. Loss Function:
   - Reconstruction loss: MSE between predicted next frame and actual next frame
   - Codebook commitment loss: To ensure encoder outputs stay close to codebook entries
   - Codebook entropy regularization: To encourage usage of full codebook

### Training Process
1. Train for 100,000 iterations with batch size 64
2. Use Adam optimizer with learning rate 3e-4, decaying to 1e-4
3. Implement codebook reset for unused embeddings (every 10,000 iterations)
4. Log reconstruction quality and codebook usage statistics
5. Save model checkpoints every 10,000 iterations

### Latent Action Extraction
1. Create a function to extract latent action indices from any pair of consecutive frames
2. Process the entire dataset to generate a new dataset of:
   - Current frame
   - Corresponding latent action index
   - Next frame
3. Save this processed dataset for use in Part 3

### Evaluation
1. Measure reconstruction quality:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
2. Analyze codebook usage:
   - Histogram of code utilization
   - Visualization of code embeddings using t-SNE
3. Qualitatively evaluate by visualizing:
   - Original frame pairs
   - Reconstructed next frames
   - Differences between actual and predicted

### Success Criteria
- Reconstruction PSNR > 30 dB
- At least 50% of codebook being actively used
- Visually plausible next frame reconstructions
- Clear patterns in latent code assignment for similar transitions