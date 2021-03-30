# -*- coding: utf-8 -*-
"""SEEDmain_code_.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i-2Z2PJEZNST8Y4caTLwB9vTIcXE3I8Y
"""

#You should delete the save files to run a new simulation. If this is not done this script will load the previous checkpoints and logs.

#!pip install tensorflow --upgrade


#!git clone https://github.com/kenjyoung/MinAtar.git
#%cd MinAtar
#!pip install .

#!apt install swig
#!pip install box2d box2d-kengz

!pip install dopamine-rl==3.1.9

import numpy as np
import os
#from dopamine.agents.dqn import dqn_agent

import dopamine
from dopamine.discrete_domains import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf


import os
import dopamine
from dopamine.discrete_domains import run_experiment
import gin.tf

import sys

import matplotlib
#matplotlib.use('TKAgg')

#import minatar
#minatar.__version__

from google.colab import drive 
drive.mount('/content/drive')
path = '/content/drive/My Drive/SaveFiles/Data/Dopamine_github/ExperimentsSeeds/Experiments/'
sys.path.append(path)


import sys
from dqn_agent_new import *
from rainbow_agent_new import *
from quantile_agent_new import *
from implicit_quantile_agent_new import *
import networks_new

agents = {
    'dqn': JaxDQNAgentNew,
    'rainbow': JaxRainbowAgentNew,
    'quantile': JaxQuantileAgentNew,
    'implicit': JaxImplicitQuantileAgentNew,
}

scale = [1.0, 1.0/jnp.sqrt(3.0), 0.1, 0.3, 0.8, 1, 2]
mode = ['fan_avg','fan_in','fan_in','fan_in','fan_in','fan_in','fan_in']
distribution=['uniform','uniform','uniform','uniform','uniform','uniform','uniform']

#0:initializers.zeros, 1:nn.initializers.ones 2:nn.initializers.variance_scaling
inits = [0, 1, 2]

num_runs = 1
environments = ['cartpole', 'acrobot']
seeds = [True, False]

for seed in seeds:
  for agent in agents:
    for env in environments:
      for init in inits:
        for i in range (1, num_runs + 1):
          print({agents[agent]})
          print(f'{agents[agent]}')        
          
          def create_agent(sess, environment, summary_writer=None):
            return agents[agent](num_actions=environment.action_space.n)

          if init == 2:
            for j in range(0,len(scale)):
              LOG_PATH = os.path.join(f'{path}{seed}{i}_{agent}_{env}__initia{init}_{scale[j]}_{mode[j]}', f'dqn_test{i}')
              sys.path.append(path)    
              gin_file = f'{path}{agent}_{env}.gin'

              agent_name = agents[agent].__name__
              gin_bindings = [f"{agent_name}.seed=None"] if seed is False else [f"{agent_name}.seed={i}",
                              f"JaxDQNAgentNew.initzer = @networks_new.variance_scaling_init()",
                              f"networks_new.variance_scaling_init.type = {init}",
                              f"networks_new.variance_scaling_init.scale = {scale[j]}"
                              f"networks_new.variance_scaling_init.mode = {mode[j]}"
                              f"networks_new.variance_scaling_init.distribution = {distribution[j]}"]
          else:
            LOG_PATH = os.path.join(f'{path}{seed}{i}_{agent}_{env}_initia_{init}', f'dqn_test{i}')
            sys.path.append(path)    
            gin_file = f'{path}{agent}_{env}.gin'

            agent_name = agents[agent].__name__
            gin_bindings = [f"{agent_name}.seed=None"] if seed is False else [f"{agent_name}.seed={i}",
                            f"JaxDQNAgentNew.initzer = @networks_new.variance_scaling_init()",
                            f"networks_new.variance_scaling_init.type = {init}"]


          gin.parse_config_files_and_bindings([gin_file], gin_bindings, skip_unknown=False)
          agent_runner = run_experiment.TrainRunner(LOG_PATH, create_agent)

          print(f'Will train agent {agent} in {env}, run {i}, please be patient, may be a while...')
          agent_runner.run_experiment()
          print('Done training!')