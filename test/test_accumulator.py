import unittest
import sys
import os
from pathlib import Path

import jax
from jax import random
import jax.numpy as jnp
import haiku as hk
import gym
import numpy as np

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
from utils import experience

def random_transition(rng_key):
  keys = random.split(rng_key, 5)
  action = random.normal(keys[0])
  step_type = random.choice(keys[1], np.array([0,1]), (), p=np.array([0.8,0.2]))
  obsv = random.normal(keys[2], (2,3,))
  reward = random.normal(keys[3])
  discount = random.normal(keys[4])
  timestep = experience.TimeStep(step_type, obsv, reward, discount)
  return action, timestep

def transition_i(i, termination=False):
  action = 1
  step_type = 1 if termination else 0
  obsv = jnp.array([[i,i,i],[i,i,i]])
  reward = 1
  discount = 0
  timestep = experience.TimeStep(step_type, obsv, reward, discount)
  return action, timestep

class TestFunctionality(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.env = gym.make('CartPole-v1')
    cls.key = random.PRNGKey(42)
    return super().setUpClass()

  def setUp(self) -> None:
    max_t = 100
    max_ep = 10
    max_transition = 1000
    self.look_back = 4
    self.acc = experience.Accumulator(max_t, max_ep, max_transition, look_back = self.look_back)
    self.no_lb_acc = experience.Accumulator(max_t, max_ep, max_transition, look_back = 1)
    return super().setUp()

  def test_init(self):
    assert(self.acc.len_ep==0)
    assert(self.acc.len_transitions==0)
    assert(self.acc._look_back_obsv.size==0)

  def test_run_rand(self):
    num_trial = 20
    batch_size = 4
    for _ in range(num_trial):
      key, rng_key = random.split(TestFunctionality.key)
      self.acc.push(*random_transition(rng_key))
      self.no_lb_acc.push(*random_transition(rng_key))
    batch_ts = self.acc.sample_batch_transtions(rng_key, batch_size)
    assert(batch_ts.s_tm1.shape==(batch_size, self.look_back, 2, 3))

  def test_run_det0(self):
    pass

  def test_run_det1(self):
    num_trial = 5
    for i in range(num_trial):
      self.acc.push(*transition_i(i))
    self.acc.push(*transition_i(num_trial, termination=True))
    assert(1==1)
    assert(self.acc.len_ep==1)
    assert(self.acc.len_transitions==num_trial)
    assert(self.acc._transitions.at(0).s_tm1.shape == (self.look_back, 2, 3))
    assert(self.acc._transitions.at(0).s_t.shape == (self.look_back, 2, 3))
    assert(sum([np.sum(v) for v in self.acc._transitions.at(0).s_tm1])==0)
    assert(sum([np.sum(v) for v in self.acc._transitions.at(0).s_t])==6)

    
    


if __name__ == '__main__':
  unittest.main()