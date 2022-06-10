import gym
import numpy as np

class Normalize(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, mean: np.ndarray, sd:np.ndarray):
        gym.ObservationWrapper.__init__(self, env)
        self.mean = mean
        self.sd = sd

    def observation(self, obsv: np.ndarray) -> np.ndarray:
        return (obsv-self.mean)/self.sd
