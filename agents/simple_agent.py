import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from functools import partial

from value_prediction import approximator, q
from value_prediction.approximator import LinearApproximator
from agents.base import Agent


class epsilonGreedyAgent(Agent):
  """
  Agent acts (behavior policy) with epsilon greedy.
  Agent learns with q learning with tabular function approximator.
  """
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

  def train_init(self, rng_key):
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

class epsLinearAgent(Agent):
  """
  Agent acts (behavior policy) with epsilon greedy.
  Agent learns with q learning with linear function approximator.

  This is deadly triad agent:
    - bootstrap               (in q_learning)
    - off policy              (q_learning use greedy policy)
    - function approximation  (linear)
  """
  def __init__(self, env, epsilon, eps_decay_rate=0.99, learning_rate=0.1, lr_decay=1):
    self.env = env
    self.state_space = env.observation_space
    self.action_space = env.action_space
    if hasattr(self.state_space, 'high'):
      # self.state_max = self.state_space.high
      self.state_max = jnp.array([4.8, 10, 0.42, 10]) # Cartpole
    else:
      self.state_max = jnp.array([sp.n-1 for sp in self.state_space])
    self.appr = LinearApproximator(len(self.state_max), self.action_space.n)
    self.appr_q_all_batch = jax.jit(jax.vmap(LinearApproximator.v, in_axes=[None, 0]))
    self.init_lr = learning_rate
    self.lr_decay = lr_decay
    self.discount=0.9
    self.init_epsilon = epsilon
    self.eps_decay_rate = eps_decay_rate
    self.episode_number = 0
    self.recent_loss=0

  def train_init(self, rng_key):
    self.appr.assign_W(random.uniform(rng_key, self.appr.W.shape)*1e-3)
    self.episode_number = 0
    self.epsilon = self.init_epsilon
    self.lr = self.init_lr

  def episode_init(self, initial_observation):
    self.episode_number+=1

  def act(self, observation, rngkey):
    q_t = LinearApproximator.v(self.appr.W, jnp.array(observation))
    argmax_a = jnp.argmax(q_t)
    rn = random.uniform(rngkey)
    action = int( (rn<self.epsilon)* self.action_space.sample() + (rn>=self.epsilon)*argmax_a)
    return action, self.discount

  def opt_step(self):
    self.episode_number+=1
    self.epsilon *= self.eps_decay_rate
    self.epsilon = max(self.epsilon, 0.1)
    self.lr *= self.lr_decay

  def eval_act(self, observation, rngkey):
    q_t = LinearApproximator.v(self.appr.W, jnp.array(observation))
    argmax_a = jnp.argmax(q_t)
    return int(argmax_a), self.discount

  def learn_one_ep(self, episode):
    a_tm1, timesteps = episode
    states = self.norm_state(timesteps.obsv)
    q_all = self.appr_q_all_batch(self.appr.W, states)
    a_tm1 = a_tm1[1:]
    s_tm1 = states[:-1]
    r_t = timesteps.reward[1:]
    # q_tm1 = q_all[:-1]
    q_t = q_all[1:]

    targets = jax.vmap(q.q_learning_target, in_axes=[0,0,None])(r_t, q_t, self.discount)
    grad, loss = LinearApproximator.batched_weight_update(self.appr.W, s_tm1, a_tm1, targets)
    self.recent_loss = loss
    assert(grad.shape ==  self.appr.W.shape)
    self.appr.assign_W(self.appr.W - self.lr*grad)
    self.opt_step()

  def norm_state(self, state):
    return state/self.state_max
