import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def td_zero_error(v_tm1, r_t, v_t, discount):
  return r_t+discount*v_t - v_tm1

@partial(jax.jit, static_argnums=(0,))
def nstep_td_errors(n : int, v_tm1, r_t, v_t, discount_t):
  """
    G(n,t) = r(t+1) + gamma*G(n-1, t+1)
    G(1,t) = r(t+1) + gamma*v(t+1)
  """
  discount_t = discount_t * jnp.ones_like(r_t) # if scalar, to array
  assert(n>0)
  assert(v_tm1.shape == r_t.shape)
  assert(v_t.shape == r_t.shape)
  assert(r_t.shape[0]>=n)

  targets = v_t[n-1:]
  for i in range(n):
    targets = r_t[n-1-i:len(r_t)-i] + targets * discount_t[n-1-i:len(discount_t)-i]
  
  return targets-v_tm1[:len(v_tm1)-n+1]

@partial(jax.jit, static_argnums=(0,))
def td_lambda_errors(lmd, v_tm1, r_t, v_t, discount_t):
  """
    G(n,t) = r(t+1) + gamma((1-lmd)v(t+1)+lmdG(t+1))
  """
  discount_t = discount_t * jnp.ones_like(r_t) # if scalar, to array
  assert(v_tm1.shape == r_t.shape)
  assert(v_t.shape == r_t.shape)

  G_t = v_t[-1]
  returns=[]
  for i in range(v_tm1.shape[0]-1,-1,-1):
    G_t = r_t[i]+discount_t[i]*((1-lmd)*v_t[i] + lmd*G_t)
    returns.append(G_t)
  return jnp.array(returns)[::-1]-v_tm1
