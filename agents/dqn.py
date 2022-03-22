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
  assert(type(args[0])==hk.Module)
  return hk.transform(lambda x: args[0](*args[1:], **kwargs)(x))

class Dqn(Agent):
  """


  * implementing `step` in `__init__` is inspired by https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/dqn/agent.py
  """
  def __init__(self, env, network : hk.Transformed, epsilon, discount=0.9, learning_rate=0.1, delay_update=10):
    self.env = env
    self.state_space = env.observation_space
    self.action_space = env.action_space
    self.network_init, self.network_apply = hk.without_apply_rng(network)
    optimizer = optax.adam(learning_rate=learning_rate)
    self.opt_init = optimizer.init

    self.discount=discount
    self.epsilon = epsilon
    self.delay_update = delay_update
    self.step_count = 0
    
    @jax.jit
    def batch_value(params, batch_state):
      return jax.vmap(network.apply, in_axes=(None, 0))(params, x=batch_state)

    @jax.jit
    def loss(replay_params, target_params, s_tm1, a_tm1, r_t, s_t):
      q_tm1 = batch_value(replay_params, s_tm1)
      q_t = batch_value(target_params, s_t)
      targets = jax.vmap(q.q_learning, in_axes=(0,0,0,0,None,None))(q_tm1, a_tm1, r_t, q_t, discount, True)
      return jnp.mean((jax.lax.stop_gradient(targets)-q_tm1)**2/2) # probably not need to stop_gradient since they are different params

    @jax.jit
    def step(replay_params, target_params, opt_state, s_tm1, a_tm1, r_t, s_t):
      loss_value, grads = jax.value_and_grad(loss, argnums=1)(replay_params, target_params, s_tm1, a_tm1, r_t, s_t)
      updates, opt_state = optimizer.update(grads, opt_state, replay_params)
      replay_params = optax.apply_updates(replay_params, updates)
      return replay_params, opt_state, loss_value

    self.batch_value = batch_value
    self.step = step

  def train_init(self, rng_key):
    self.target_params = self.network_init(rng=rng_key, x=jnp.zeros((len(self.state_space),)))
    self.replay_params = self.target_param
    self.opt_state = self.opt_init(self.replay_param)
    self.step_count = 0


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
    pass

  def learn_batch_transitions(self, transitions):
    s_tm1 = transitions.s_tm1
    a_tm1 = transitions.a_tm1
    r_t = transitions.r_t
    s_t = transitions.s_t
    
    self.replay_param, self.opt_state, loss =self.step(self.replay_param,
                                                        self.target_param,
                                                        self.opt_state,
                                                        s_tm1,
                                                        a_tm1,
                                                        r_t,
                                                        s_t)
    self.step_count +=1
    if not self.step_count%self.delay_update:
      self.target_param=self.replay_param
