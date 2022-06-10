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
    self.acc_temp = experience.Accumulator(max_t, max_ep, max_transition, look_back = self.look_back)
    self.no_lb_acc = experience.Accumulator(max_t, max_ep, max_transition, look_back = 1)
    return super().setUp()

  def test_init(self):
    self.assertTrue(self.acc.len_ep==0)
    self.assertTrue(self.acc.len_transitions==0)
    self.assertTrue(self.acc._look_back_obsv.size==0)

  def test_run_rand(self):
    num_trial = 20
    batch_size = 4
    for _ in range(num_trial):
      key, rng_key = random.split(TestFunctionality.key)
      self.acc.push(*random_transition(rng_key))
      self.no_lb_acc.push(*random_transition(rng_key))
    batch_ts = self.acc.sample_batch_transtions(rng_key, batch_size)
    self.assertTrue(batch_ts.s_tm1.shape==(batch_size, 2, self.look_back*3))

  def test_run_det0(self):
    pass

  def test_run_det1(self):
    num_trial = 5
    for i in range(num_trial):
      self.acc.push(*transition_i(i))
    self.acc.push(*transition_i(num_trial, termination=True))
    self.assertTrue(self.acc.len_ep==1)
    self.assertTrue(self.acc.len_transitions==num_trial)
    self.assertTrue(self.acc._transitions.at(0).s_tm1.shape == (2, self.look_back*3))
    self.assertTrue(self.acc._transitions.at(0).s_t.shape == (2, self.look_back*3))
    self.assertTrue(sum([np.sum(v) for v in self.acc._transitions.at(0).s_tm1])==0)
    self.assertTrue(sum([np.sum(v) for v in self.acc._transitions.at(0).s_t])==6)

  def test_reproducibility(self):
    num_trial = 20
    batch_size = 2
    key, rng_key = random.split(TestFunctionality.key)
    for _ in range(num_trial):
      key, rng_key = random.split(key)
      arg = random_transition(rng_key)
      self.acc.push(*arg)
      self.acc_temp.push(*arg)
    for _ in range(5):
      key, rng_key = random.split(key)
      bt1 = self.acc.sample_batch_transtions(rng_key, batch_size)
      bt2 = self.acc_temp.sample_batch_transtions(rng_key, batch_size)
      ac1, ep1 = self.acc.sample_one_ep(rng_key = rng_key)
      ac2, ep2 = self.acc_temp.sample_one_ep(rng_key = rng_key)
      for v in zip([bt1.s_tm1,ac1,ep1.obsv],[bt2.s_tm1,ac2,ep2.obsv]):
        self.assertTrue(v[0].shape==v[1].shape)
        self.assertTrue(all(jnp.ravel(jnp.equal(jnp.array(v[0].astype(np.float32)), jnp.array(v[1].astype(np.float32))))))


    
    


if __name__ == '__main__':
  unittest.main()