import unittest
import sys
import os
from pathlib import Path

import jax
import gym
import numpy as np

# sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
# from utils import experience, experiment
# from value_prediction import approximator
# from agents.dqn import MLP_TargetNetwork, get_transformed
# from agents import *

class TestTest(unittest.TestCase):
  def test_initialization(self):
    self.assertEqual(1,1)


if __name__ == '__main__':
  unittest.main()