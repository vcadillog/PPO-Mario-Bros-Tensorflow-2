import numpy as np
import os
from multiprocessing import Process, Pipe
import random
import Common_constants 
import Auxiliars as AUX

env = Common_constants.env
gae_lambda = Common_constants.gae_lambda
gamma = Common_constants.gamma
max_steps = Common_constants.max_steps
horizon = Common_constants.horizon
start_t = Common_constants.start_t

class SubProcessEnv(object):
    def env_process(name, conn):
        # env = make_train_0()
        # env = make_atari(name)        
        # env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True)
        
#         env = gym.make(name)
        while True:
            (cmd, args, kwargs) = conn.recv()
            if cmd == "reset":
                conn.send(env.reset())
            elif cmd == "step":
                conn.send(env.step(*args, **kwargs))
            elif cmd == "exit":
                break
            else:
                raise Exception("Unknown command %s" % (str(cmd),))

    def __init__(self, name):
        parent_conn, child_conn = Pipe()
        self.process = Process(target=SubProcessEnv.env_process, args=(name, child_conn))
        self.process.start()
        self.conn = parent_conn

    def step(self, action):
        self.conn.send(("step", (action,), {}))
        return self.conn.recv()

    def reset(self):
        self.conn.send(("reset", (), {}))
        return self.conn.recv()

    def exit(self):
        self.conn.send(("exit", (), {}))
        self.process.join()

class TimeIndexedList(object):
    def __init__(self, first_t=0):
        self.first_t = first_t
        self.list = []

    # For flushing so that we don't keep unnecessary history forever.
    def flush_through(self, t):
        to_remove = t - self.first_t + 1
        if to_remove > 0:
            self.list = self.list[to_remove:]
            self.first_t = t + 1

    def append(self, elem):
        self.list.append(elem)

    def get(self, t):
        return self.list[t - self.first_t]

    def future_length(self):
        return len(self.list)

    def get_range(self, t, length):
        return self.list[(t - self.first_t):(t - self.first_t + length)]

# Data relevant to executing and collecting samples from a single environment execution.
class EnvActor(object):
    def __init__(self, env):
        self.env = env
        self.obs = TimeIndexedList(first_t = start_t)
        self.last_obs = self.env.reset()
        self.last_obs = np.expand_dims(self.last_obs , axis = 0)     
        self.last_obs = tf.convert_to_tensor(self.last_obs, dtype=tf.float32)     # for performance
        self.obs.append(self.last_obs)
        # self.pobs = TimeIndexedList(first_t = start_t)
        # self.last_pobs = preprocess_obs_atari(self.obs, self.pobs, start_t, start_t)
        # self.pobs.append(self.last_pobs)
        self.act = TimeIndexedList(first_t = start_t)
        self.rew = TimeIndexedList(first_t = start_t)
        self.val = TimeIndexedList(first_t = start_t)
        self.policy = TimeIndexedList(first_t = start_t)
        self.delta = TimeIndexedList(first_t = start_t)
        self.done = TimeIndexedList(first_t = start_t)
        self.episode_start_t = 0
        self.episode_rewards = []
        self.episode_x = []
        self.rewards_this_episode = []
        self.x_this_episode = []
        self.advantage_estimates = TimeIndexedList(first_t = start_t)
        self.value_estimates = TimeIndexedList(first_t = start_t)

    def step_env(self,policy_net,value_net,t):      
        if t == start_t:
            # Artifact of ordering             
            val_0 = value_net(self.last_obs).numpy()[0]
            self.val.append(val_0[0])
        
        policy_t = policy_net(self.last_obs).numpy()[0]                
        # policy_t = tf.nn.softmax(policy_t, axis = 1)[0]        
        # print(policy_t)
        action_t = np.random.choice(num_actions, 1, p=policy_t)[0]        

        
        obs_tp1,rew_t,done_t,info_t = self.env.step(action_t)        
        obs_tp1 = np.expand_dims(obs_tp1,axis = 0)

        self.act.append(action_t)
        self.rew.append(rew_t)
        self.policy.append(policy_t)
        self.rewards_this_episode.append(rew_t)        
        self.x_this_episode.append(info_t.get('x_pos')) 

        if done_t:
            self.done.append(True)
            self.episode_rewards.append(sum(self.rewards_this_episode))
            self.rewards_this_episode = []
            self.episode_x.append(self.x_this_episode)
            self.x_this_episode = []
            obs_tp1 = self.env.reset()
            obs_tp1 = np.expand_dims(obs_tp1,axis = 0)
            self.episode_start_t = t + 1
        else:
            self.done.append(False)

        # Important to put this after we've updated obs_tp1 in case of reset.
        # NOTE: Bug fix, obs_horizon was being added to before the possible reset, so wrong observation was associated to initial policy step.
        self.obs.append(obs_tp1)
        
        # pobs_tp1 = preprocess_obs_atari(self.obs, self.pobs, t + 1, self.episode_start_t)
        # self.pobs.append(pobs_tp1)       
        obs_tp1 = tf.convert_to_tensor(obs_tp1, dtype=tf.float32)     # for performance 
        val_tp1 = value_net(obs_tp1).numpy()[0]       
        self.val.append(val_tp1[0])  
        self.delta.append(self.rew.get(t) +  (1 - AUX.indicator(done_t)) * gamma * self.val.get(t + 1) - self.val.get(t))       
        self.last_obs = obs_tp1        

    # end_t is non-inclusive, i.e. it's the t immediately after the desired horizon.
    def calculate_horizon_advantages(self, end_t):
        advantage_estimates = []
        value_estimates = []
        advantage_so_far = 0
        # No empirical estimate beyond end of horizon, so use value function.  Is immediately reset to 0 if at episode boundary.
        last_value_sample = self.val.get(end_t)
        for ii in range(horizon):
            if self.done.get(end_t - ii - 1):
                advantage_so_far = 0
                last_value_sample = 0
            
            # Insert in reverse order.
            advantage_so_far = self.delta.get(end_t - ii - 1) + (gamma * gae_lambda * advantage_so_far)
            advantage_estimates.append(advantage_so_far)
            # NOTE: Was using 1-step value update here; instead use the GAE value estimate (i.e. Q(s,a) with the empirical action.)
            #last_value_sample = (1 - indicator(self.done.get(end_t - ii - 1))) * gamma * last_value_sample + self.rew.get(end_t - ii - 1)
            # Didn't need the 1 - indicator since setting this above.
            last_value_sample = gamma * last_value_sample + self.rew.get(end_t - ii - 1)
            value_estimates.append(last_value_sample)
            #value_estimates.append(advantage_so_far + self.val.get(end_t - ii - 1))
            
            #value_sample_estimates.append((1 - indicator(done_horizon.get(t - ii - 1))) * gamma * val_horizon.get(t - ii)  + rew_horizon.get(t - ii - 1)) 
        advantage_estimates.reverse()
        value_estimates.reverse()

        # NOTE: Was normalizing here, but moved that to whole batch.
        for ii in range(len(advantage_estimates)):
            self.advantage_estimates.append(advantage_estimates[ii])
            self.value_estimates.append(value_estimates[ii])


    def get_horizon(self, end_t):
        return (self.obs.get_range(end_t - horizon, horizon),
                self.act.get_range(end_t - horizon, horizon),
                self.policy.get_range(end_t - horizon, horizon),
                self.advantage_estimates.get_range(end_t - horizon, horizon),
                self.value_estimates.get_range(end_t - horizon, horizon))

    def flush(self, end_t):
        # Retain some extra observations for preprocessing step.
        self.obs.flush_through(end_t - horizon - 5)
        self.act.flush_through(end_t - horizon - 1)
        self.rew.flush_through(end_t - horizon - 1)
        self.val.flush_through(end_t - horizon - 1)
        self.policy.flush_through(end_t - horizon - 1)
        self.delta.flush_through(end_t - horizon - 1)
        self.done.flush_through(end_t - horizon - 1)
