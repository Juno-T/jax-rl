import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from functools import partial
import haiku as hk
import optax

from value_prediction import q
from agents.base import Agent

class MLP_TargetNetwork(hk.Module):
  """
  Just use hk.nets.MLP honestly.
  """
  def __init__(self, output_sizes, name='MLP_TargetNetwork'):
    super().__init__(name=name)
    self._internal_linear = hk.nets.MLP(output_sizes, name='internal_linear')

  def __call__(self, x):
    return self._internal_linear(x)

def get_transformed(*args, **kwargs):
  return hk.transform(lambda x: args[0](*args[1:], **kwargs)(x))

class BarebonesDqn(Agent):
  """


  * implementing `step` in `__init__` is inspired by https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/dqn/agent.py
  """
  def __init__(self, env, network : hk.Transformed, epsilon, eps_decay_rate=0.99, discount=0.9, learning_rate=0.1, delay_update=10):
    self.env = env
    self.state_space = env.observation_space
    self.action_space = env.action_space
    self.network_init, self.network_apply = hk.without_apply_rng(network)
    optimizer = optax.adam(learning_rate=learning_rate)
    self.opt_init = optimizer.init
    self.eps_decay_rate = eps_decay_rate

    self.discount=discount
    self.init_epsilon = epsilon
    self.delay_update = delay_update
    self.step_count = 0
    
    @jax.jit
    def batch_value(params, batch_state):
      return jax.vmap(network.apply, in_axes=(None, None, 0))(params, None, batch_state)

    @jax.jit
    def loss(replay_params, target_params, s_tm1, a_tm1, r_t, s_t):
      q_tm1 = batch_value(replay_params, s_tm1)
      q_t = batch_value(target_params, s_t)
      targets = jax.vmap(q.q_learning, in_axes=(0,0,0,0,None,None))(q_tm1, a_tm1, r_t, q_t, discount, True)
      qa_indices = tuple(jnp.stack((jnp.arange(len(a_tm1)),a_tm1), axis=0))
      return jnp.mean((jax.lax.stop_gradient(targets)-q_tm1[qa_indices])**2/2) # probably don't need to stop_gradient since they are different params

    @jax.jit
    def step(replay_params, target_params, opt_state, s_tm1, a_tm1, r_t, s_t):
      loss_value, grads = jax.value_and_grad(loss, argnums=1)(replay_params, target_params, s_tm1, a_tm1, r_t, s_t)
      updates, opt_state = optimizer.update(grads, opt_state, replay_params)
      replay_params = optax.apply_updates(replay_params, updates)
      return replay_params, opt_state, loss_value

    self.batch_value = batch_value
    self.step = step

  def train_init(self, rng_key):
    self.target_params = self.network_init(rng=rng_key, x=jnp.zeros(self.state_space.shape))
    self.replay_params = self.target_params
    self.opt_state = self.opt_init(self.replay_params)
    self.step_count = 0
    self.epsilon = self.init_epsilon


  def episode_init(self, initial_observation):
    pass

  def act(self, observation, rngkey):
    q_t = self.network_apply(self.replay_params, x=observation)
    argmax_a = jnp.argmax(q_t)
    rn = random.uniform(rngkey)
    action = int( (rn<self.epsilon)* self.action_space.sample() + (rn>=self.epsilon)*argmax_a)
    return action, self.discount

  def eval_act(self, observation, rngkey=None):
    q_t = self.network_apply(self.replay_params, x=observation)
    argmax_a = jnp.argmax(q_t)
    return int(argmax_a), self.discount

  def write(self, writer, episode_number):
    writer.add_scalar('train/epsilon', self.epsilon, episode_number)
    writer.add_scalar('train/loss', self.recent_loss, episode_number)

  def learn_batch_transitions(self, transitions):
    s_tm1 = self.norm_state(transitions.s_tm1)
    a_tm1 = transitions.a_tm1
    r_t = transitions.r_t
    s_t = self.norm_state(transitions.s_t)
    
    self.replay_params, self.opt_state, self.recent_loss =self.step(self.replay_params,
                                                        self.target_params,
                                                        self.opt_state,
                                                        s_tm1,
                                                        a_tm1,
                                                        r_t,
                                                        s_t)
    self.step_count +=1
    self.epsilon *= self.eps_decay_rate
    if not self.step_count%self.delay_update:
      self.target_params=self.replay_params

  def norm_state(self, state):
    return state/jnp.array([4.8, 10, 0.42, 10]) # Cartpole
