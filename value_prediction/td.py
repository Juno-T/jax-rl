import numpy as np
import jax
import jax.numpy as jnp

def td_zero_error(v_tm1, r_t, v_t, discount):
  return r_t+discount*v_t - v_tm1

def nstep_td_errors(n, v_tm1, r_t, v_t, discount):
  """
    G(n,t) = rt + gamma*G(n-1, t+1)
    G(1,t) = rt + gamma*v(t+1)
  """
  targets = v_t[n-1:]
  for i in range(n):
    targets = r_t[n-1-i:len(r_t)-i] + targets * discount
  
  return targets-v_tm1[:len(v_tm1)-n+1]
