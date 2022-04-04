# RL algorithms with JAX

Hands-on project after finishing [Deepmind's RL course](https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm). Decided to learn and use JAX for the implementations.  

:construction: The project is still in progress.

## TODO

**Utilities**:
- Experience accumulator
  - :heavy_check_mark: by episodes
  - :black_square_button: by transitions (:construction: in progress)
- :heavy_check_mark: Training experiment

  
**Environments**:
- gym
  - :heavy_check_mark: Black-jack
  - :heavy_check_mark: Cartpole
  - :black_square_button: Atari (:construction: in progress)
- mujoco ?
- [evogym](https://github.com/EvolutionGym/evogym) (so cool, must try)

**Algorithms**:
- Value function approximator
  - :heavy_check_mark: Tabular
  - :heavy_check_mark: Linear
  - :heavy_check_mark: Neural Nets
- Value approximation/heuristic
  - TD
    - :heavy_check_mark: TD(0)
    - :heavy_check_mark: n-step TD
    - :heavy_check_mark: TD(λ)
  - Q-learning
    - :heavy_check_mark: vanilla q-learning
    - :black_square_button: λ q-learning
- Simple agents
  - :heavy_check_mark: Tabular + TD,Q (with ε-greedy)
  - :heavy_check_mark: Linear + Q (with ε-greedy)
- DQN
  - :heavy_check_mark: Barebones (NN + Q)
  - :black_square_button: [Vanilla DQN](https://arxiv.org/abs/1312.5602v1) (:construction: in progress)
  - :black_square_button: Rainbow?
- Policy Gradient
  - :black_square_button: Vanilla
  - :black_square_button: Trust Region/PPO?
- Model-based ?
- GVF ?
- Combining with Evolutionary ?
