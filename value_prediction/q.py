import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def q_learning(q_tm1, a_tm1, r_t, q_t, discount_t):
  """
  target = r_t + max(q_t)
  """

  target = r_t + discount_t * jnp.max(q_t)
  return target - q_tm1[a_tm1]

