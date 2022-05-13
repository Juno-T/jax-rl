import unittest
import sys
import os
from pathlib import Path

import jax
import haiku as hk
import jax.numpy as jnp
import jax
import gym
import numpy as np

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
from utils import experience, experiment
from value_prediction import approximator
from agents.dqn import MLP_TargetNetwork, DQN_CNN, get_transformed
from agents import *

class TestReproducibility(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.env = gym.make('CartPole-v1')
    cls.key = random.PRNGKey(42)
    return super().setUpClass()

  def setUp(self) -> None:
    config = {
      'eps_decay_rate':1-3e-3, 
      'learning_rate': .01,
      'delay_update':100
    }

    tn = get_transformed(MLP_TargetNetwork, output_sizes= [10, TestReproducibility.env.action_space.n])
    epsilon = 1
    self.agent1 = BarebonesDqn(TestReproducibility.env, 
                      tn, 
                      epsilon, 
                      eps_decay_rate=config['eps_decay_rate'], 
                      learning_rate=config['learning_rate'],
                      delay_update=config['delay_update'])
    self.agent2 = BarebonesDqn(TestReproducibility.env, 
                      tn, 
                      epsilon, 
                      eps_decay_rate=config['eps_decay_rate'], 
                      learning_rate=config['learning_rate'],
                      delay_update=config['delay_update'])
    return super().setUp()

  def test_initialization(self):
    self.agent1.train_init(TestReproducibility.key)
    self.agent2.train_init(TestReproducibility.key)
    iseq_pyt = jax.tree_util.tree_map(lambda a,b: jnp.isclose(a,b), self.agent1.target_params, self.agent2.target_params)
    self.assertTrue(all(jax.tree_util.tree_flatten(iseq_pyt)))
    iseq_pyt = jax.tree_util.tree_map(lambda a,b: jnp.isclose(a,b), self.agent1.replay_params, self.agent2.replay_params)
    self.assertTrue(all(jax.tree_util.tree_flatten(iseq_pyt)))
    iseq_pyt = jax.tree_util.tree_map(lambda a,b: jnp.isclose(a,b), self.agent1.opt_state, self.agent2.opt_state)
    self.assertTrue(all(jax.tree_util.tree_flatten(iseq_pyt)))
    self.assertTrue(all([
      self.agent1.epsilon == self.agent2.epsilon,
      self.agent1.step_count == self.agent2.step_count,
    ]))
    # self.agent1.episode_init()
    # self.agent2.episode_init()

  def test_action(self):
    self.agent1.train_init(TestReproducibility.key)
    self.agent2.train_init(TestReproducibility.key)

    act_root_rngkey, sample_rngkey, eval_rngkey = random.split(TestReproducibility.key, 3)
    observation = jnp.array(TestReproducibility.env.reset(seed=np.random.randint(0,1e5)))
    self.agent1.episode_init(observation)
    self.agent2.episode_init(observation)
    
    done = False
    while not done:
      act_root_rngkey, act_rngkey = random.split(act_root_rngkey)
      action1, discount = self.agent1.act(observation, act_rngkey)
      action2, discount = self.agent1.act(observation, act_rngkey)
      self.assertEqual(action1, action2)
      observation, reward, done, info = self.env.step(action1)
    
class TestGeneral(unittest.TestCase):
  def test_network_call(self):
    out_size = 10
    batch_size = 14
    key = random.PRNGKey(42)
    init, apply = hk.transform(lambda x: DQN_CNN(out_size)(x))
    params = init(rng = key, x=jax.random.normal(key, (4,84,84)))
    sample_x = jax.random.normal(key, (batch_size, 4, 84, 84))
    out = apply(params=params, x=sample_x, rng=key)
    self.assertTrue(out.shape==(batch_size, out_size))


if __name__ == '__main__':
  unittest.main()