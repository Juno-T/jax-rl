from abc import ABC, abstractmethod
import time

import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from value_prediction import approximator, td

class Agent(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def train_init(self, rng_key):
    pass


  @abstractmethod
  def episode_init(self, initial_observation):
    pass

  @abstractmethod
  def act(self, observation, rngkey=None):
    pass

  def eval_act(self, observation, rngkey=None):
    return self.act(observation, rngkey)

  def get_stats(self):
    raise("Not implemented")

  def learn_one_ep(self, episode):
    raise("Not implemented")

  def learn_batch_transitions(self, transitions):
    raise("Not implemented")

class RandomAgent(Agent):
  def __init__(self, env, learning_rate=0.1):
    self.env = env
    self.state_space = env.observation_space
    self.action_space = env.action_space
    self.discount=0.9
    

  def train_init(self, rng_key):
    pass

  def episode_init(self, initial_observation):
    pass

  def act(self, observation, rngkey):
    rand_action = random.randint(rngkey, (), 0, self.action_space.n)
    return rand_action, self.discount

  # @partial(jax.jit, static_argnums=(0,))
  def learn_one_ep(self, episode):
    pass

  def get_stats(self):
    pass

  def learn_batch_transitions(self, transitions):
    pass