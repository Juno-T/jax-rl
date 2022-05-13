from abc import ABC, abstractmethod
import time

import numpy as np
import jax
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
    self.appr = approximator.tabularApproximator([np.arange(sp.n) for sp in self.state_space])
    self.appr_v_batch = jax.jit(jax.vmap(self.appr.v))
    self.lr = learning_rate
    self.discount=0.9
    

  def train_init(self, rng_key):
    self.appr.init_table(jnp.zeros_like(self.appr.table))

  def episode_init(self, initial_observation):
    pass

  def act(self, observation, rngkey=None):
    return self.action_space.sample(), self.discount

  # @partial(jax.jit, static_argnums=(0,))
  def learn_one_ep(self, episode):
    
    time_laps=[]
    time_laps.append(time.time())
    a_tm1, timesteps = episode
    # v_all = jax.vmap(self.appr.v)(timesteps.obsv)
    v_all = self.appr_v_batch(timesteps.obsv)
    s_tm1 = timesteps.obsv[:-1]
    r_t = timesteps.reward[1:]
    v_tm1 = v_all[:-1]
    v_t = v_all[1:]
    errors = td.td_lambda_errors(.5, v_tm1, r_t, v_t, self.discount)
    self.appr.batch_update(s_tm1, errors*self.lr + v_tm1)