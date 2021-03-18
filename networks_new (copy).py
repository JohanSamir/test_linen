"""Various networks for Jax Dopamine agents."""

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import gym_lib
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
from jax import random
import math

from jax.tree_util import tree_flatten, tree_map

#---------------------------------------------------------------------------------------------------------------------


env_inf = {"CartPole":{"MIN_VALS": onp.array([-2.4, -5., -math.pi/12., -math.pi*2.]),"MAX_VALS": onp.array([2.4, 5., math.pi/12., math.pi*2.])},
            "Acrobot":{"MIN_VALS": onp.array([-1., -1., -1., -1., -5., -5.]),"MAX_VALS": onp.array([1., 1., 1., 1., 5., 5.])},
            "MountainCar":{"MIN_VALS":onp.array([-1.2, -0.07]),"MAX_VALS": onp.array([0.6, 0.07])}
            }

initializers = {"xavier_uniform": nn.initializers.xavier_uniform(), 
                "variance_scaling": jax.nn.initializers.variance_scaling(scale=1.0/jnp.sqrt(3.0),mode='fan_in',distribution='uniform')}
#---------------------------------------------------------------------------------------------------------------------

class NoisyNetwork(nn.Module):
  features: int
  rng: int
  bias_in: bool

  @nn.compact
  def __call__(self, x):

    def sample_noise(rng_input, shape):
      noise = jax.random.normal(rng_input,shape)
      return noise

    def f(x):
      return jnp.multiply(jnp.sign(x), jnp.power(jnp.abs(x), 0.5))
    # Initializer of \mu and \sigma 
   
    def mu_init(key, shape, rng):
        low = -1*1/jnp.power(x.shape[1], 0.5)
        high = 1*1/jnp.power(x.shape[1], 0.5)
        return jax.random.uniform(rng, shape=shape, minval=low, maxval=high)

    def sigma_init(key, shape, dtype=jnp.float32): return jnp.ones(shape, dtype)*(0.1 / jnp.sqrt(x.shape[1]))

    rng, rng2, rng3, rng4, rng5 = jax.random.split(self.rng, 5)
    # Sample noise from gaussian
    p = sample_noise(rng2,[x.shape[1], 1])
    q = sample_noise(rng3,[1, self.features])
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = jnp.squeeze(f_q)
    w_mu = self.param('kernel', mu_init, (x.shape[1], self.features), rng3)
    w_sigma = self.param('kernell', sigma_init, (x.shape[1], self.features))
    w = w_mu + jnp.multiply(w_sigma, w_epsilon)
    ret = jnp.matmul(x, w)

    b_mu = self.param('bias', mu_init, (self.features,), rng5)
    b_sigma = self.param('biass',sigma_init, (self.features,))
    b = b_mu + jnp.multiply(b_sigma, b_epsilon)
    return jnp.where(self.bias_in, ret + b, ret)
  
#---------------------------------------------< DQNNetwork >----------------------------------------------------------

@gin.configurable
class DQNNetwork(nn.Module):
  """Jax DQN network for Cartpole."""
  num_actions:int
  net_conf: str
  env: str
  normalize_obs:bool
  noisy: bool
  dueling: bool
  initzer:str
  hidden_layer: int
  neurons: int

  @nn.compact
  def __call__(self, x , rng):

    if self.net_conf == 'minatar':
      x = x.squeeze(3)
      x = x[None, ...]
      x = x.astype(jnp.float32)
      x = nn.Conv(features=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),  kernel_init=initializers["variance_scaling"])(x)
      x = jax.nn.relu(x)
      x = x.reshape((x.shape[0], -1))

    elif self.net_conf == 'atari':
      # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
      # have removed the true batch dimension.
      x = x[None, ...]
      x = x.astype(jnp.float32) / 255.
      x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                  kernel_init=initializers["variance_scaling"])(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                  kernel_init=initializers["variance_scaling"])(x)
      x = jax.nn.relu(x)
      x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                  kernel_init=initializers["variance_scaling"])(x)
      x = jax.nn.relu(x)
      x = x.reshape((x.shape[0], -1))  # flatten

    elif self.net_conf == 'classic':
      #classic environments
      x = x[None, ...]
      x = x.astype(jnp.float32)
      x = x.reshape((x.shape[0], -1))

    if self.env is not None and self.env in env_inf:
      x = x - env_inf[self.env]['MIN_VALS']
      x /= env_inf[self.env]['MAX_VALS'] - env_inf[self.env]['MIN_VALS']
      x = 2.0 * x - 1.0

    if self.noisy:
      def net(x, features, rng):
        return NoisyNetwork(features, rng=rng, bias_in=True)(x)
    else:
      def net(x, features, rng):
        return nn.Dense(features, kernel_init=initializers["variance_scaling"])(x)

    for _ in range(self.hidden_layer):
      x = net(x, features=self.neurons, rng=rng)
      x = jax.nn.relu(x)

    adv = net(x, features=self.num_actions, rng=rng)
    val = net(x, features=1, rng=rng)
    dueling_q = val + (adv - (jnp.mean(adv, -1, keepdims=True)))
    non_dueling_q = net(x, features=self.num_actions, rng=rng)

    q_values = jnp.where(self.dueling, dueling_q, non_dueling_q)

    return atari_lib.DQNNetworkType(q_values)