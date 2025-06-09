### Why this project?
minimal dataset prep + fastest path to test world-model ideas, had very limited time <12hrs
[thread of expirement xlog](https://x.com/k7agar/status/1931807294830071912)

### More compute/time?
Action-conditioned Diffusion Transformer over multiple games or try to test limits of what physics of world models could be learned via VQ-VAE for robotic world modeling.
Testing [flow matching (x link for expirements)](https://x.com/k7agar/status/1929438412953838062) for more effecent and accurate prediction.

### What did you learn?
primary bottleneck in experiments was the pixel reconstruction process, was time-consuming.  the learning of latent dynamics was very efficient and has many low hanging fruits that can be grabbed re: flow mathcing and computaional optmisation,  pixel-level details require significant resources to process, the underlying latent representations of the dynamics are much more straightforward for the model to learn as the world model is truly very very simple for this case.

### Biggest surprise?
Predictor loss flat-lines almost instantly—Pong physics is trivial once latent is good. scaling laws for game world modeling and how progressing the data and the steps lead to latent learning of game physics and engine

### Paper-ready extras?
Ablations on latent size/λ, action-conditioned control tasks, flow matching, world modelling on hardware, even more beautiful expirements, better computational sampling methods to optimise the trianing and infernece of world models for real time perf gains


