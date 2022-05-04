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

  def __init__(self, env, accumulator:Accumulator, logdir, onEpisodeSummary = lambda step, data: None):
    self.env = env
    self.acc = accumulator
    # TODO: clean legacy tensorboard logger
    self.writer = SummaryWriter(logdir)
    self.trained_ep = 0
    self.onEpisodeSummary = onEpisodeSummary
    
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
      rngkey, env_rngkey, act_root_rngkey, sample_rngkey, eval_rngkey = random.split(rngkey, 5)
      episode_summary = {'train': {}, 'val':{}, 'agent': {}}


      observation = jnp.array(self.env.reset(seed=int(random.randint(env_rngkey, (), 0, 1e5))))
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
      train_reward = jnp.sum(timesteps.reward).item()
      episode_summary['train']['reward']=train_reward
      self.writer.add_scalar('train/reward', train_reward, episode_number)

      if learn_from_transitions:
        transitions = self.acc.sample_batch_transtions(sample_rngkey, batch_size)
        agent.learn_batch_transitions(transitions)
      else:
        agent.learn_one_ep(episode)
      
      if episode_number%evaluate_every==0:
        test_reward = self.eval(eval_rngkey, agent, eval_episodes, episode_number)
        episode_summary['val']['reward'] = test_reward
      episode_summary['agent']=agent.get_stats()
      agent.write(self.writer, episode_number)

      self.onEpisodeSummary(episode_number, episode_summary)
    self.trained_ep += train_episodes
      

  def eval(self, rngkey, agent, eval_episodes, episode_number):
    for ep in range(eval_episodes):
      rngkey, env_rngkey, act_root_rngkey = random.split(rngkey, 3)
      observation = jnp.array(self.env.reset(seed=int(random.randint(env_rngkey, (), 0, 1e5))))
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
    test_reward = jnp.sum(rewards).item()
    self.writer.add_scalar('eval/reward', test_reward,episode_number)
    return test_reward