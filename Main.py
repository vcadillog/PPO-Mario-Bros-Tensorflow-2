from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import random
import cv2
import tensorflow as tf
import NeuralNets as NN
import PPO
import Datapreprocessing as dpp
import Enviroments as Env
import Common_constants as CC
import MultiEnv as ME
import Auxiliars as AUX

tf.keras.backend.set_floatx('float32')

env = CC.env
obs_shape = CC.obs_shape
num_actions = CC.num_actions
env_name = CC.env_name
start_t = CC.start_t
save_path = CC.save_path
num_actors = CC.num_actors
max_steps = CC.max_steps
base_learning_rate = CC.base_learning_rate
log_dir = CC.log_dir
SMALL_NUM = CC.SMALL_NUM
load_model = CC.load_model

def train(load = True):
    
    writer_sum = tf.summary.create_file_writer(log_dir)
    # Global time counter
    # At time t, take a_t from s_t, receive r_t.
    t = start_t
    last_save = 0
    
    actors = []
    value_network = NN.value_nn()
    policy_network = NN.policy_nn()    
 
    if load:
      AUX.loader([value_network,policy_network],save_path)
      
    for ii in range(num_actors):
        actors.append(ME.EnvActor(ME.SubProcessEnv(env_name, env)))

    while t <= max_steps:
        learning_rate = base_learning_rate * ME.alpha_anneal(t)        
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)   
        model_grads = PPO.gradients(adam)

        for ii in range(horizon):
            for actor in actors:                                
                actor.step_env(policy_network,value_network,t)
            t += 1

        for actor in actors:

            actor.calculate_horizon_advantages(t)

        # Construct randomly sampled (without replacement) mini-batches.
        obs_horizon = []
        act_horizon = []
        policy_horizon = []
        adv_est_horizon = []
        val_est_horizon = []

        for actor in actors:
            obs_a, act_a, policy_a, adv_est_a, val_est_a = actor.get_horizon(t)
            obs_horizon.extend(obs_a)
            act_horizon.extend(act_a)
            policy_horizon.extend(policy_a)
            adv_est_horizon.extend(adv_est_a)
            val_est_horizon.extend(val_est_a)

        # Normalizing advantage estimates.
        # NOTE:  Adding this significantly improved performance
        # NOTE: Moved this out of each individual actor, so that advantages for the whole batch are normalized with each other.
        adv_est_horizon = np.array(adv_est_horizon)
        adv_est_horizon = (adv_est_horizon - np.mean(adv_est_horizon)) / (np.std(adv_est_horizon) + SMALL_NUM)

        num_samples = len(obs_horizon)
        indices = list(range(num_samples))

        for e in range(optim_epochs):
            random.shuffle(indices)
            ii = 0
        #     # TODO: Don't crash if batch_size is not a divisor of total sample count.
            while ii < num_samples:
                obs_batch = []
                act_batch = []
                policy_batch = []
                adv_batch = []
                value_sample_batch = []

                for b in range(batch_size):
                    index = indices[ii]                    
                    obs_batch.append(np.squeeze(obs_horizon[index],axis=0))
                    act_batch.append(act_horizon[index])
                    policy_batch.append(policy_horizon[index])
                    adv_batch.append(adv_est_horizon[index])
                    value_sample_batch.append(val_est_horizon[index])
                    ii += 1

                # Training loop
                obs_batch = tf.convert_to_tensor(np.asarray(obs_batch), dtype=tf.float32) 
                act_batch = tf.convert_to_tensor(np.asarray(act_batch), dtype=tf.uint8) 
                policy_batch = tf.convert_to_tensor(np.asarray(policy_batch), dtype=tf.float32) 
                adv_batch = tf.convert_to_tensor(np.asarray(adv_batch), dtype=tf.float32) 
                value_sample_batch = tf.convert_to_tensor(np.asarray(value_sample_batch), dtype=tf.float32)    

                #Calling Function of training                
                entropy_loss, clip_loss, value_loss, total_loss = model_grads(AUX.alpha_anneal(t),
                                                                    policy_network, value_network,obs_batch,
                                                                    act_batch,adv_batch,policy_batch,value_sample_batch)
                
           
        for actor in actors:
            actor.flush(t)

        if t - last_save > 1000:
            AUX.saver([value_network,policy_network],save_path)
            last_save = t

        all_ep_rewards = []
        all_ep_x = []
        for actor in actors:
            all_ep_rewards.extend(actor.episode_rewards)
            all_ep_x.extend(actor.episode_x)
        if len(all_ep_rewards) >= 10:
            print("T: %d" % (t,))
            print("AVG Reward: %f" % (np.mean(all_ep_rewards),))
            print("MIN Reward: %f" % (np.amin(all_ep_rewards),))
            print("MAX Reward: %f" % (np.amax(all_ep_rewards),))            
            print("AVG X: %f" % (np.hstack(all_ep_x).mean(),))
            print("MIN X: %f" % (np.hstack(all_ep_x).min(),))
            print("MAX X: %f" % (np.hstack(all_ep_x).max(),))            
            AUX.sum_writer(writer_sum,np.mean(all_ep_rewards),t,'Avg_Reward')
            AUX.sum_writer(writer_sum,np.hstack(all_ep_x).mean(),t,'Avg_X')
            AUX.sum_writer(writer_sum,np.hstack(all_ep_x).max(),t,'Max_X')
            for actor in actors:
                actor.episode_rewards = []
                actor.episode_x = []
            # print("Entropy Loss: %f" % (np.mean(entropy_loss),))
            # print("Value Loss: %f" % (np.mean(value_loss),))
            # print("Clip Loss: %f" % (np.mean(clip_loss),))
            # print("Total Loss: %f" % (np.mean(total_loss),))

def test(episodes,env_test):

    assert load_model == True
    assert env_test >= 0 and env_test<=3

    env_test = Env.make_env(env_test)
    policy_network = NN.policy_nn()
    AUX.loader_test(policy_network,save_path)
    done = False
    scores = []
    for e in range(episodes):

        state = env_test.reset()
        state = np.expand_dims(state , axis = 0)     
        state = tf.convert_to_tensor(state, dtype=tf.float32)     # for performance      
        score = 0
        video_frames = []
        while True:    

            video_frames.append(cv2.cvtColor(env_test.render(mode = 'rgb_array'), cv2.COLOR_RGB2BGR))
            policy_t = policy_network(state).numpy()[0]                            
            action_t = np.argmax(policy_t) # Deterministic action            
            state,reward,done,_ = env_test.step(action_t)        
            state = np.expand_dims(state,axis = 0)
            state = tf.convert_to_tensor(state, dtype=tf.float32)     # for performance       
            score += reward 

            if done:
                break

        video_name = 'test_' + str(e)+'.mp4'
        _, height, width, _ = np.shape(video_frames)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(video_name, fourcc, 5, (width,height))
        for image in video_frames:            
            video.write(image)

        cv2.destroyAllWindows()
        video.release()        
        print('Test #%s , Score: %0.1f' %(e, score))    
        scores.append(score)


    print('Average reward: %0.2f of %s episodes' %(np.mean(scores),episodes))            
