# Part 5: Playable World Model Interface

## Overview
In this final part, we'll integrate all previous components to create a playable Breakout game that runs entirely through our learned world model. This will demonstrate the effectiveness of our approach and provide an interactive way to evaluate the model.

## Related Components
- [part2.md](part2.md): Provides the latent action codebook
- [part3.md](part3.md): Provides the next frame prediction model
- [part4.md](part4.md): Provides the action-to-latent mapping

## Implementation Details

### Game State Representation
1. Maintain the current frame as the game state
2. Initialize with a real starting frame from the dataset
3. Track additional metadata:
   - Score (extracted from frame via simple image processing)
   - Lives remaining (extracted from frame)
   - Ball position (tracked for debugging)

### Game Loop Implementation
1. Display current frame to the player
2. Capture player input (LEFT, RIGHT, NOTHING)
3. Convert player action to latent action using mapping from Part 4
4. Use next frame prediction model from Part 3 to generate the next frame
5. Update game state with the predicted frame
6. Repeat at 15-20 FPS for smooth gameplay

### User Interface
1. Create a simple PyGame interface:
   - Main game window showing current frame (scaled up for visibility)
   - Score display
   - Lives remaining
   - Optional debug info (predicted latent code, certainty, etc.)
2. Controls:
   - Left/Right arrow keys for paddle movement
   - Space to pause/resume
   - R to reset game
   - D to toggle debug information

### Performance Optimization
1. Implement frame caching for common states
2. Run neural network inference on GPU
3. Use asynchronous prediction (predict next frames in background)
4. Maintain consistent frame rate with frame skipping if needed

### Reality Anchoring (Preventing Drift)
1. Detect and correct physically impossible states:
   - Ball moving through bricks
   - Ball disappearing
   - Paddle moving too quickly
2. Implement simple heuristics for corrections
3. Optional: Implement occasional "reality anchoring" by resetting to a similar known state

### Evaluation and User Testing
1. Invite 3-5 players to try the game and provide feedback
2. Record gameplay sessions for analysis
3. Metrics to track:
   - Average score achieved
   - Average game duration
   - Number of obvious visual glitches
   - Player satisfaction ratings

### Extended Features (If Time Permits)
1. Multi-step planning:
   - Show predicted future frames based on current action
   - Help player make strategic decisions
2. Model exploration:
   - UI to browse and visualize latent action effects
   - Heatmap of next frame prediction confidence
3. Comparison mode:
   - Side-by-side play with real Breakout game
   - Visual diff highlighting prediction errors

### Success Criteria
- Game runs at stable 15+ FPS
- Ball physics remains consistent for extended play
- Paddle responds correctly to player inputs
- Brick destruction works properly
- Game is actually playable and somewhat enjoyable