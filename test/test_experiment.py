import unittest
import sys
import os
from pathlib import Path
import gym

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
from utils import experience, experiment
from value_prediction import approximator
from agents.dqn import MLP_TargetNetwork, get_transformed
from agents import *

class TestReproducibility(unittest.TestCase):

  def setUp(self) -> None:
    return super().setUp()
  
  def test_env(self):
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



if __name__ == '__main__':
  unittest.main()