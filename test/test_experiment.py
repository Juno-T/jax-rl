import unittest
import sys
import os
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
import gym
import numpy as np
import stable_baselines3 as sb3

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
from utils import experience, experiment
from agents import RandomAgent

class TestReproducibility(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.key = random.PRNGKey(42)
    return super().setUpClass()

  def setUp(self) -> None:
    return super().setUp()
  
  def test_gym_CartPole(self):
    self.env1 = gym.make('CartPole-v1')
    self.env2 = gym.make('CartPole-v1')
    for i in range(10):
      reset_seed= np.random.randint(0,1e5)
      observation1 = jnp.array(self.env1.reset(seed=reset_seed))
      observation2 = jnp.array(self.env2.reset(seed=reset_seed))
      self.assertTrue(all(jnp.equal(observation1, observation2)))

      done = False
      while not done:
        action = self.env1.action_space.sample()
        observation1, reward1, done, info = self.env1.step(action)
        observation2, reward2, _, _ = self.env2.step(action)
        self.assertTrue(all(jnp.equal(observation1, observation2)))
        self.assertEqual(reward1, reward2)

  def test_wrapped_atari(self):
    gym_env1 = gym.make('ALE/BeamRider-v5')
    gym_env2 = gym.make('ALE/BeamRider-v5')
    self.env1 = sb3.common.atari_wrappers.AtariWrapper(gym_env1, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    self.env2 = sb3.common.atari_wrappers.AtariWrapper(gym_env2, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    for i in range(5):
      reset_seed= np.random.randint(0,1e5)
      observation1 = jnp.array(self.env1.reset(seed=reset_seed))
      observation2 = jnp.array(self.env2.reset(seed=reset_seed))
      self.assertTrue(all(jnp.ravel(jnp.equal(observation1, observation2))))

      done = False
      while not done:
        action = self.env1.action_space.sample()
        observation1, reward1, done, info = self.env1.step(action)
        observation2, reward2, _, _ = self.env2.step(action)
        self.assertTrue(all(jnp.ravel(jnp.allclose(observation1, observation2))))
        self.assertEqual(reward1, reward2)

  def test_Trainer(self):
    acc1 = experience.Accumulator(501,3,10000, look_back=4)
    acc2 = experience.Accumulator(501,3,10000, look_back=4)
    gym_env1 = gym.make('ALE/BeamRider-v5')
    gym_env2 = gym.make('ALE/BeamRider-v5')
    self.env1 = sb3.common.atari_wrappers.AtariWrapper(gym_env1, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    self.env2 = sb3.common.atari_wrappers.AtariWrapper(gym_env2, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    self.ab_agent1 = RandomAgent(self.env1)
    self.ab_agent2 = RandomAgent(self.env2)
    
    ep_sum1=None
    ep_sum2=None
    def onEpisodeSummary1(step, ep_sum):
      ep_sum1 = ep_sum

    def onEpisodeSummary2(step, ep_sum):
      ep_sum2 = ep_sum
    
    trainer1 = experiment.Trainer(self.env1, acc1, onEpisodeSummary=onEpisodeSummary1)
    trainer2 = experiment.Trainer(self.env2, acc2, onEpisodeSummary=onEpisodeSummary2)

    train_episodes = 2
    key, train_key = random.split(TestReproducibility.key)
    trainer1.train(train_key, self.ab_agent1, train_episodes, batch_size=3, is_continue=False, learn_from_transitions=True)
    trainer2.train(train_key, self.ab_agent2, train_episodes, batch_size=3, is_continue=False, learn_from_transitions=True)

    self.assertTrue(trainer1.trained_ep == trainer2.trained_ep)
    action1, timesteps1 = trainer1.acc.sample_one_ep(rng_key=key)
    action2, timesteps2 = trainer2.acc.sample_one_ep(rng_key=key)
    self.assertTrue(action1.shape==action2.shape)
    self.assertTrue(timesteps1.obsv.shape==timesteps2.obsv.shape)
    self.assertTrue(all(jnp.equal(action1, action2)))
    self.assertTrue(all(jnp.ravel(jnp.equal(timesteps1.obsv, timesteps2.obsv))))

    if(ep_sum1 is not None) and (ep_sum2 is not None):
      self.assertTrue(ep_sum1['train']['reward'] == ep_sum2['train']['reward'])
      self.assertTrue(ep_sum1['val']['reward'] == ep_sum2['val']['reward'])


class TestTrainClass():

  @classmethod
  def setUpClass(cls) -> None:
    cls.key = random.PRNGKey(42)
    return super().setUpClass()

  def setUp(self) -> None:
    self.cartpole = gym.make('CartPole-v1')
    gym_env = gym.make('ALE/BeamRider-v5')
    self.atari_beam = sb3.common.atari_wrappers.AtariWrapper(gym_env, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    self.cp_agent = RandomAgent(self.cartpole)
    self.ab_agent = RandomAgent(self.atari_beam)
    return super().setUp()
  
  def test_Train_train(self):
    acc = experience.Accumulator(501,3,10000)
    trainer = experiment.Trainer(self.cartpole, acc, onEpisodeSummary=onEpisodeSummary)

    train_episodes = 10
    key, train_key = random.split(TestTrainClass.key)
    trainer.train(train_key, self.cp_agent, train_episodes, batch_size=100, is_continue=False, learn_from_transitions=True)
    assert(1==1)
    



if __name__ == '__main__':
  unittest.main()