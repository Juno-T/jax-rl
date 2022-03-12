import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


class tabularApproximator:
  def __init__(self, state_range):
    self._state_range = [jnp.sort(jnp.array(r)) for r in state_range]
    self.table = jnp.zeros([len(r) for r in state_range], dtype=jnp.float32)


  def init_table(self, table):
    """ 
      Manually init table
    """
    assert(self.table.shape == table.shape)
    self.table = table

  def v(self, state):
    """
    """
    # assert(len(state) == len(self._state_range))
    return self.table[tuple(self.indexof(state))]

  # @partial(jax.jit, static_argnums=(0,))
  def indexof(self, state):
    idx = []
    for i in range(len(state)):
      idx.append(jnp.searchsorted(self._state_range[i], state[i]))
    return idx

  def update(self, state, value):
    self.table = self.table.at[tuple(self.indexof(state))].set(value)
    return 0

  def batch_update(self, states, values):
    
    for s, v in zip(states, values):
      self.update(s,v)
    # @jax.jit
    # def _batch_update_table(table, states, values):
    #   states = jnp.array(states)
    #   values = jnp.array(values)
    #   def _update_table(table, x):
    #     return table.at[tuple(self.indexof(x[:-1]))].set(x[-1]), None
    #   table, ys = jax.lax.scan(_update_table, \
    #     table, jnp.concatenate((states, jnp.expand_dims(values, axis=1)),axis=1))
    #   return table
    # self.table = _batch_update_table(self.table, states, values)

class tabularQApproximator(tabularApproximator):
  def __init__(self, state_range, action_range):
    try:
      _ = len(action_range)
    except:
      action_range = [action_range]
    super().__init__(list(state_range)+list(action_range))

  def q(self, state, action):
    return super().v(self.parse_state_action(state, action))

  def q_all(self, state):
    return super().v(state)


  def update(self, state, action, value):
    return super().update(self.parse_state_action(state,action), value)

  def batch_update(self, states, actions, values):
    for args in zip(states, actions, values):
      self.update(*args)
      
  def parse_state_action(self, state, action):
    """
    state: array-like
    action: array-like
    """
    if action is None:
      return state
    try:
      _ = len(action)
    except:
      action = [action]
    return jnp.concatenate((jnp.array(state), jnp.array(action)), axis=0)