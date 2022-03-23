# RL algorithms with JAX

Hands-on project after finishing [Deepmind's RL course](https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm). Decided to learn and use JAX for the implementations.  

:construction: The project is still in progress.

## TODO

**Utilities**:
- Experience accumulator
  - [x] by episodes
  - [ ] by transitions (:construction: in progress)
- [x] Training experiment

  
**Environments**:
- gym
  - [x] Black-jack
  - [x] Cartpole
  - [ ] Atari ?
- mujoco ?
- [evogym](https://github.com/EvolutionGym/evogym) (so cool, must try)

**Algorithms**:
- Value function approximator
  - [x] Tabular
  - [x] Linear
  - [ ] Neural Nets (:construction: in progress)
- Value approximation/heuristic
  - TD
    - [x] TD(0)
    - [x] n-step TD
    - [x] TD($\lambda$)
  - Q-learning
    - [x] vanilla q-learning
    - [ ] $\lambda$ q-learning
- Simple agents
  - [X] Tabular + TD,Q (with $\epsilon$-greedy)
  - [X] Linear + Q (with $\epsilon$-greedy)
- DQN
  - [ ] Barebones (NN + Q) (:construction: in progress)
  - [ ] Vanilla DQN
  - [ ] Rainbow?
- Policy Gradient
  - [ ] Vanilla
  - [ ] Trust Region/PPO?
- Model-based ?
- GVF ?
- Combining with Evolutionary ?
