import collections
from typing import NamedTuple, Any

import numpy as np
import jax
from jax import random

class TimeStep(NamedTuple):
  step_type: int = 0  # is terminal step
  obsv: Any = None
  reward: float = 0.0
  discount: float = 1.0


# Modified from Deepmind/rlax example
class Accumulator:
  """Accumulator for gathering episodes.
    Could be used as an online accumulator or as an experience storage.

    Init:
      max_t: maximum number of timesteps to store
      max_ep: maximum number of episodes to store

    Attributes:
      _episodes: deque stores timesteps (episode)
      _timesteps: sequence of (action, timestep)

    Method:
      _new_episode
      push
      sample_one_ep
      len_ep
      has_new_ep


  """

  def __init__(self, max_t, max_ep):
    self._episodes = collections.deque(maxlen=max_ep)
    self._max_t = max_t
    self._prev_ep_len = 0
    self._new_episode()

  def _new_episode(self):
    self._timesteps = collections.deque(maxlen = self._max_t)

  def push(self, action : Any, timestep: TimeStep):
    # Replace `None`s with zeros as these will be put into NumPy arrays.
    a_tm1 = 0 if action is None else action
    timestep_t = timestep._replace(
        step_type=int(timestep.step_type),
        reward=0. if timestep.reward is None else timestep.reward,
        discount=0. if timestep.discount is None else timestep.discount,
    )
    self._timesteps.append((a_tm1, timestep_t))
    if timestep_t.step_type: # terminal
      self._episodes.append(self._timesteps)
      self._new_episode()
    
  def sample_one_ep(self, current=False, previous=False, rng_key=None):
    if current:
      actions, timesteps = jax.tree_multimap(lambda *ts: np.stack(ts),
                                           *self._timesteps)
    
    elif previous:
      assert(len(self._episodes)>=1)
      actions, timesteps = jax.tree_multimap(lambda *ts: np.stack(ts),
                                           *self._episodes[-1])
    else:
      assert(len(self._episodes)>=1)
      assert(rng_key is not None)
      ep_idx = random.randint(rng_key, (), 0, len(self._episodes))
      actions, timesteps = jax.tree_multimap(lambda *ts: np.stack(ts),
                                           *self._episodes[ep_idx])
    return actions, timesteps

  def len_ep(self):
    return len(self._episodes)

  def has_new_ep(self):
    if len(self._episodes)>self._prev_ep_len:
      self._prev_ep_len=len(self._episodes)
      return True
    return False

  # def sample(self, batch_size):
  #   """Returns current sequence of accumulated timesteps."""
  #   if batch_size != 1:
  #     raise ValueError("Require batch_size == 1.")
  #   if len(self._timesteps) != self._timesteps.maxlen:
  #     raise ValueError("Not enough timesteps for a full sequence.")

  #   actions, timesteps = jax.tree_multimap(lambda *ts: np.stack(ts),
  #                                          *self._timesteps)
  #   return actions, timesteps

  # def is_ready(self, batch_size):
  #   if batch_size != 1:
  #     raise ValueError("Require batch_size == 1.")
  #   return len(self._timesteps) == self._timesteps.maxlen