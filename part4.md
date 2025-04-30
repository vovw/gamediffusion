# Part 4: Action Mapping Model

## Overview
In this part, we'll create a model that maps the actual game controls (LEFT, RIGHT, NOTHING) to the latent action space we learned in Part 2. This will allow us to control the world model with normal game inputs.

## Related Components
- [part1.md](part1.md): Provides the original action labels
- [part2.md](part2.md): Provides the latent action codes
- [part5.md](part5.md): Uses this mapping to convert user inputs into world model updates

## Implementation Details

### Data Preparation
1. For each frame transition in our dataset:
   - Get the original action (LEFT, RIGHT, NOTHING) from Part 1 recordings
   - Get the corresponding latent action code predicted by our model from Part 2
   - Create pairs of (original_action, latent_action_code)
2. Create a balanced dataset by sampling equally from each action type
3. Split into train/validation sets (80%/20%)

### Model Architecture
1. Implement a simple MLP classifier:
   - Input: One-hot encoded original action (3 dimensions)
   - Hidden layers: 512 → 256 → 128 neurons
   - Output: Softmax over all possible latent action codes (e.g., 256 classes)
   - ReLU activations and dropout (0.2) between hidden layers

2. Alternative approach (if above doesn't work well):
   - Create separate probability distributions for each action type
   - When an action is selected, sample from its specific distribution
   - This allows for one-to-many mapping (one action can cause different latent actions)

### Training Process
1. Train for 20,000 iterations with batch size 128
2. Use Adam optimizer with learning rate 1e-3
3. Cross-entropy loss for classification
4. Track accuracy on validation set
5. Save model checkpoints every 5,000 iterations

### Evaluation and Analysis
1. Compute confusion matrix between predicted and actual latent codes
2. Calculate accuracy per action type
3. Visualize the distribution of latent codes for each action
4. Qualitative analysis:
   - For each action, show examples of frame transitions it produces
   - Identify any inconsistencies or unexpected mappings

### Inference Pipeline
1. Create a simple function that:
   - Takes a game action as input
   - Returns the most likely latent action code
   - Alternatively, samples from the probability distribution
2. Test this pipeline by feeding actions and visualizing resulting transitions

### Success Criteria
- Classification accuracy > 80%
- Clear visual distinction between effects of different actions
- Paddle movements correctly corresponding to LEFT/RIGHT actions
- NOTHING action correctly mapped to appropriate latent codes