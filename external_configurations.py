"""External configuration .gin"""

from gin import config
import jax

config.external_configurable(jax.nn.initializers.zeros, 'jax.nn.initializers.zeros')
config.external_configurable(jax.nn.initializers.ones, 'jax.nn.initializers.ones')
config.external_configurable(jax.nn.initializers.variance_scaling, 'jax.nn.initializers.variance_scaling')
