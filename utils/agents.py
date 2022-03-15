from abc import ABC, abstractmethod
import collections
from typing import NamedTuple, Any
import time

import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from functools import partial

from value_prediction import approximator, td, q

class Agent(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def train_init(self):
    pass


  @abstractmethod
  def episode_init(self, initial_observation):
    pass

  @abstractmethod
  def act(self, observation, rngkey=None):
    pass

  @abstractmethod
  def learn_one_ep(self, episode):
    pass

class RandomAgent(Agent):
  def __init__(self, env, learning_rate=0.1):
    self.env = env
    self.state_space = env.observation_space
    self.action_space = env.action_space
    self.appr = approximator.tabularApproximator([np.arange(sp.n) for sp in self.state_space])
    self.appr_v_batch = jax.jit(jax.vmap(self.appr.v))
    self.lr = learning_rate
    self.discount=0.9
    

  def train_init(self):
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

class epsilonGreedyAgent(Agent):
  def __init__(self, env, epsilon, learning_rate=0.1):
    self.env = env
    self.state_space = env.observation_space
    self.action_space = env.action_space
    self.appr = approximator.tabularQApproximator(
        [np.arange(sp.n) for sp in self.state_space],
        [np.arange(self.action_space.n)],
        ordered_state = True)
    self.appr_q_all_batch = jax.jit(jax.vmap(self.appr.q_all))
    self.lr = learning_rate
    self.discount=0.9
    self.epsilon = epsilon

  def train_init(self):
    self.appr.init_table(jnp.zeros_like(self.appr.table))

  def episode_init(self, initial_observation):
    pass


  def act(self, observation, rngkey):
    q_t = self.appr.q(observation, None)
    argmax_a = jnp.argmax(q_t)
    rn = random.uniform(rngkey)
    return int( (rn<self.epsilon)* self.action_space.sample() + (rn>=self.epsilon)*argmax_a), self.discount

  def learn_one_ep(self, episode):
    a_tm1, timesteps = episode
    q_all = self.appr_q_all_batch(timesteps.obsv)
    a_tm1 = a_tm1[1:]
    s_tm1 = timesteps.obsv[:-1]
    r_t = timesteps.reward[1:]
    q_tm1 = q_all[:-1]
    q_t = q_all[1:]
    @partial(jax.vmap, in_axes=(0,0,0,0,None, None))
    def _batched_q_learning_update(*args):
      return q.q_learning(*args[:5])*args[5]+args[0][args[1]]
    updated = _batched_q_learning_update(q_tm1, a_tm1, r_t, q_t, self.discount, self.lr)
    # error = jax.vmap(q.q_learning, in_axes=(0,0,0,0,None))(q_tm1, a_tm1, r_t, q_t, self.discount)
    # qa_indices = tuple(jnp.stack((jnp.arange(len(a_tm1)),a_tm1), axis=0))
    # updated = error*self.lr + q_tm1[qa_indices]
    self.appr.batch_update(s_tm1, a_tm1, updated)