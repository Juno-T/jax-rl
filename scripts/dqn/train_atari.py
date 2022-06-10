import gym
import stable_baselines3 as sb3
from jax import random
import sys
import os
from pathlib import Path
# import wandb

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent.parent))
from utils import experience, experiment, wrapper
from value_prediction import approximator
from agents.dqn import MLP_TargetNetwork, get_transformed
from agents import *

def onEpisodeSummary(step, data):
  pass
  # wandb.log(data=data, step=step)

def main():
  config = {
    'eps_decay_rate':1-3e-3, 
    'learning_rate': 1e-2,
    'delay_update': 200
  }
  # wandb.init(
  #   entity="yossathorn-t",
  #   project="jax-rl_dqn",
  #   notes="Test barebones dqn on gym's cartpole",
  #   tags=["dqn", "vanilla", "atari"],
  #   config=config  
  # )

  key = random.PRNGKey(42)
  
  env = gym.make('ALE/BeamRider-v5')
  env = sb3.common.atari_wrappers.AtariWrapper(env, 
                        noop_max=30, 
                        frame_skip=4, 
                        screen_size=84, 
                        terminal_on_life_loss=True, 
                        clip_reward=True) # (84,84,1)
  # env = wrapper.Transpose(env, axes=(2,0,1)) # (1,84,84)
  print(f"State space {env.action_space}, size {env.action_space.n}")
  tn = get_transformed(DQN_CNN, output_size= env.action_space.n)
  epsilon = 1
  look_back = 4
  agent = BarebonesDqn(env, 
                      tn, 
                      epsilon, 
                      look_back = look_back,
                      eps_decay_rate=config['eps_decay_rate'], 
                      learning_rate=config['learning_rate'],
                      delay_update=config['delay_update'])
  acc = experience.Accumulator(501,3,10000, look_back = look_back)
  trainer = experiment.Trainer(env, acc, onEpisodeSummary=onEpisodeSummary)

  train_episodes = 1000
  key, train_key = random.split(key)
  trainer.train(train_key, agent, train_episodes, batch_size=100, is_continue=False, learn_from_transitions=True)


  



if __name__=='__main__':
  main()