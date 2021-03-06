import numpy as np
import os
import dopamine
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
import sys

import matplotlib
from dqn_agent_new import *
from rainbow_agent_new import *
from quantile_agent_new import *
from implicit_quantile_agent_new import *
import networks_new
import external_configurations

agents = {
    'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    'quantile': JaxQuantileAgentNew,
    'implicit': JaxImplicitQuantileAgentNew,
}

inits = {
    'zeros': {'function':jax.nn.initializers.zeros},
    'ones': {'function':jax.nn.initializers.ones},
    'xavier': {'function':jax.nn.initializers.variance_scaling, 'scale':1, 'mode':'fan_avg', 'distribution':'uniform'},
    'variance_baseline':{'function':jax.nn.initializers.variance_scaling, 'scale':1.0/jnp.sqrt(3.0), 'mode':'fan_in', 'distribution':'uniform'},
    'variance_0.1':{'function':jax.nn.initializers.variance_scaling, 'scale':0.1, 'mode':'fan_in', 'distribution':'uniform'},
    'variance_0.3':{'function':jax.nn.initializers.variance_scaling, 'scale':0.3, 'mode':'fan_in', 'distribution':'uniform'},
    'variance_0.8':{'function':jax.nn.initializers.variance_scaling, 'scale':0.8, 'mode':'fan_in', 'distribution':'uniform'},
    'variance_1':{'function':jax.nn.initializers.variance_scaling, 'scale':1, 'mode':'fan_in', 'distribution':'uniform'},
    'variance_2':{'function':jax.nn.initializers.variance_scaling, 'scale':2, 'mode':'fan_in', 'distribution':'uniform'},
    'variance_5':{'function':jax.nn.initializers.variance_scaling, 'scale':5, 'mode':'fan_in', 'distribution':'uniform'},
    'variance_10':{'function':jax.nn.initializers.variance_scaling, 'scale':10, 'mode':'fan_in', 'distribution':'uniform'}
}

num_runs = 7
environments = ['cartpole', 'acrobot','lunarlander','mountaincar']
seeds = [True, False]

for seed in seeds:
  for agent in agents:
    for env in environments:
      for init in inits:
        for i in range (1, num_runs + 1):  
          
          def create_agent(sess, environment, summary_writer=None):
            return agents[agent](num_actions=environment.action_space.n)

          agent_name = agents[agent].__name__
          initializer = inits[init]['function'].__name__

          LOG_PATH = os.path.join(f'{path}{seed}{i}_{agent}_{env}_{init}', f'dqn_test{i}')
          sys.path.append(path)    
          gin_file = f'{path}{agent}_{env}.gin'

          if init == 'zeros' or init == 'ones':
            gin_bindings = [f"{agent_name}.seed=None"] if seed is False else [f"{agent_name}.seed={i}",
                            f"{agent_name}.initzer = @{initializer}"]
          else:
            mode = '"'+inits[init]['mode']+'"'
            distribution = '"'+inits[init]['distribution']+'"'
            gin_bindings = [f"{agent_name}.seed=None"] if seed is False else [f"{agent_name}.seed={i}",
                            f"{agent_name}.initzer = @{initializer}()",
                            f"{initializer}.scale = 1",
                            f"{initializer}.mode = {mode}",
                            f"{initializer}.distribution = {distribution}"
                            ]

          gin.clear_config()
          gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
          agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

          print(f'Will train agent {agent} in {env}, run {i}, please be patient, may be a while...')
          agent_runner.run_experiment()
          print('Done training!')
print('Finished!')