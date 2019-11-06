from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import Datapreprocessing as dpp

def make_env(env_idx):
  dicts=[
      {'state':'SuperMarioBros-1-1-v0'},
      {'state':'SuperMarioBros-1-2-v0'},
      {'state':'SuperMarioBros-1-3-v0'},
      {'state':'SuperMarioBros-2-2-v0'},      
  ]
  env=gym_super_mario_bros.make(id=dicts[env_idx]['state'])
  env=JoypadSpace(env,COMPLEX_MOVEMENT)

  env = dpp.EpisodicLifeEnv(env)
  env = dpp.RewardScaler(env)
  env = dpp.PreprocessFrame(env)
  env = dpp.StochasticFrameSkip(env,4,0.5)
  env = dpp.ScaledFloatFrame(env)
  env = dpp.FrameStack(env, 4)
  
  return env

