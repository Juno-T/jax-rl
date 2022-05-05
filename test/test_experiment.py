import unittest
import sys
import os
from pathlib import Path
import gym
import stable_baselines3 as sb3

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
from utils import experience, experiment
from value_prediction import approximator
from agents.dqn import MLP_TargetNetwork, get_transformed
from agents import *

class TestReproducibility(unittest.TestCase):

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




if __name__ == '__main__':
  unittest.main()