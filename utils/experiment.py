import collections
from typing import NamedTuple, Any

import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.experience import TimeStep, Accumulator
from agents import Agent
import time


class Trainer:
  """
    Interaction between agent and environment
  """

  def __init__(self, env, accumulator:Accumulator, logdir):
    self.env = env
    self.acc = accumulator
    self.writer = SummaryWriter(logdir)
    self.trained_ep = 0
    
  def _reset(self):
    pass #TODO

  def train(self, 
    rngkey, 
    agent: Agent,
    train_episodes: int, 
    batch_size: int=10, 
    evaluate_every: int=2, 
    eval_episodes: int=1, 
    is_continue: bool=False, 
    learn_from_transitions: bool=False):

    rngkey, init_rngkey = random.split(rngkey)
    if not is_continue:
      self._reset()
      self.trained_ep=0
      agent.train_init(init_rngkey)
    for episode_number in tqdm(range(self.trained_ep, self.trained_ep+train_episodes), bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}'):
      rngkey, act_root_rngkey, sample_rngkey, eval_rngkey = random.split(rngkey, 4)

      observation = jnp.array(self.env.reset())
      self.acc.push(None, TimeStep(obsv = observation))
      done=False
      agent.episode_init(observation)
      while not done:
        act_root_rngkey, act_rngkey = random.split(act_root_rngkey)
        action, discount = agent.act(observation, act_rngkey)
        observation, reward, done, info = self.env.step(action)
        self.acc.push(action, TimeStep(
            int(done),
            jnp.array(observation),
            reward,
            discount
        ))
      
      episode = self.acc.sample_one_ep(last_episode=True)
      a_tm1, timesteps = episode
      self.writer.add_scalar('train/reward', jnp.sum(timesteps.reward).item(), episode_number)

      if learn_from_transitions:
        transitions = self.acc.sample_batch_transtions(sample_rngkey, batch_size)
        agent.learn_batch_transitions(transitions)
      else:
        agent.learn_one_ep(episode)
      
      if episode_number%evaluate_every==0:
        self.eval(eval_rngkey, agent, eval_episodes, episode_number)
      agent.write(self.writer, episode_number)
    self.trained_ep += train_episodes
      

  def eval(self, rngkey, agent, eval_episodes, episode_number):
    for ep in range(eval_episodes):
      rngkey, act_root_rngkey = random.split(rngkey)
      observation = jnp.array(self.env.reset())
      # self.acc.push(None, TimeStep(obsv = observation))
      done=False
      agent.episode_init(observation)
      rewards = []
      while not done:
        act_root_rngkey, act_rngkey = random.split(act_root_rngkey)
        action, discount = agent.eval_act(observation, act_rngkey)
        observation, reward, done, info = self.env.step(action)
        rewards.append(reward)
    rewards = jnp.array(rewards)
    # todo: plot with policy entropy/ explained variance (PPO)
    self.writer.add_scalar('eval/reward',jnp.sum(rewards).item(),episode_number)