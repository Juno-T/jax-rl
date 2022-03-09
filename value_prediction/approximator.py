import numpy as np
import jax
import jax.numpy as jnp


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
    assert(len(state) == len(self._state_range))
    return self.table[tuple(self.indexof(state))]

  def indexof(self, state):
    idx = []
    for i in range(len(state)):
      idx.append(jnp.searchsorted(self._state_range[i], state[i]))
    return idx

  def update(self, state, value):
    self.table = self.table.at[tuple(self.indexof(state))].set(value)
    return 0

  

  def batch_update(self, states, values):
    def _batch_update_table(table, states, values):
      states = jnp.array(states)
      values = jnp.array(values)
      def _update_table(table, x):
        return table.at[tuple(self.indexof(x[:-1]))].set(x[-1]), None
      table, ys = jax.lax.scan(_update_table, \
        table, jnp.concatenate((states, jnp.expand_dims(values, axis=1)),axis=1))
      return table
    self.table = jax.jit(_batch_update_table)(self.table, states, values)

