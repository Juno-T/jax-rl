import numpy as np
import jax
import jax.numpy as jnp


class tabularApproximator:
  def __init__(self, state_range):
    self.state_range = [jnp.sort(jnp.array(r)) for r in state_range]
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
    assert(len(state) == len(self.state_range))
    return self.table[tuple(self.indexof(state))]

  def indexof(self, state):
    idx = []
    for i in range(len(state)):
      idx.append(jnp.searchsorted(self.state_range[i], state[i]))
    return idx

