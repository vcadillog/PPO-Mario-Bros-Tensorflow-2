# PPO-Mario-Bros-Tensorflow-2
A modular implementation for Proximal Policy Optimization in Tensorflow 2 using Eagerly Execution for the Super Mario Bros enviroment.

![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/SI_3DSVC_SuperMarioBros_image1600w.jpg)
## Requeirements:
- Tensorflow 2
- OpenCV
- OpenAI gym
- Super Mario Bros NES, developed by Kautenja

## Installing:
Clone the repository,

Change the path to the cloned repository

```
import os
os.chdir('./PPO-Mario-Bros-Tensorflow-2')
```

For training, run:
```
python -c 'from Main import train; train(True)'
```
The argument of training enables the load of weights of the trained model.

For testing the model:
```
python -c 'from Main import test; test(10,0)'
```

Where the first argument of test is the number of episodes to test the model, and the second is the number of the enviroment to test.

For the code the enviroments available are the next ones:
```
0 : SuperMarioBros-1-1-v0
The first level of the first world
1 : SuperMarioBros-1-2-v0 
The second level of the first world
2 : SuperMarioBros-1-3-v0
The third level of the first world
3 : SuperMarioBros-2-2-v0
The second level of the second world
```

To change the enviroments, modify the Enviroments.py file.

### NOTEBOOK FOR EASY STARTING 


There's an easy example of how to use this repo, in the EXAMPLE_OF_USE.ipynb notebook for Google Colab, just download it and upload to colab, there's not need to have python installed in your machine, the generated videos are in the branch of gloned repo.
Or open this link:

https://colab.research.google.com/drive/16xgJeXjteuw3WNVfHtp_t_VsXLQyumGa


### RESULTS


Eight actors were trained in the first level of Mario, and this is how it learned to finish it.

![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/mario.gif)

A plot how the average reward evolved vs the time steps, the model was trained in four steps due ethernet connection, the reward isn't the same as the raw output of Kautenja's implementation, it was previously scaled for this model, all the data pre processing is in the Datapreprocessing.py file.

![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/log1.PNG)![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/log2.PNG)![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/log3.PNG)![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/log4.PNG)

In the logs directory you can find two more plots, for average X_position and Max_X_position. 

Testing in not observed enviroments:

![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/test_2.gif) ![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/test_3.gif) ![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/test_4.gif)

### About the files of the repository:

* The Main.py file contains the train and test functions for the model.
    1. The train function saves the weights of the model every 1000 timesteps, also creates summary files to visualize the change of the average total reward, the average of the x position and the max value of x position. The load of weights is True by default.
    
    2. The test function loads the weights of the model and test in the selected levels with deterministic actions, the train do stochastic actions to encourage to the agent to explore and avoid getting stucked in a local optimal; and creates in MP4 videos of how the agent did as many of defined numbers of test was selected.

* The Common_constants.py file contains all the parameters needed for tune the algorithm, it transfer the parameters across the other files, also calls the Enviroment.py file to create the enviroment.

* The Enviroment.py file defines the enviroment of four diferent levels of Super Mario Bros and calls the preprocessing functions.

* The Datapreprocessing.py file creates several Classes to do:  
    
    1. Reset the enviroment after dying, this gives an additional negative reward of 50.
    
    2. Reset the enviroment after getting the flag or completing the level, this adds a positive reward of 100.
    
    3. Scalation of the reward, by a 0.05 factor.
    
    4. Resize the image and grayscaling for a faster performance of the neural network.
    
    5. Stochastic skipping of frames, based on [2], to add a randomness to the enviroment.
    
    6. Stacking of frames to create a sense of movement, based on the Atari DeepMind's implementation.
    
    7. Scaling the pixels of the image with 255 to get a range of [0-1] values. 
    
* The Auxiliars.py file contains some common function to use in the program, like saving, loading models.

* The MultiEnv.py file create a callable with multiple Proccess to create several actors, and also calcules the advantage estimator defined in [1].

* The PPO.py file contains tf functions to calculate the total loss defined in [1] and run gradients in eagerly execution of tensorflow 2.

* The NeuralNets.py file contains two classes of models, for the actor and the critic.
  

### This code was inspired from:

* [1] Proximal Policy Optimization Algorithms. 

  https://arxiv.org/pdf/1707.06347.pdf

* [2] Gotta Learn Fast: A New Benchmark for Generalization in RL.

  https://arxiv.org/pdf/1804.03720.pdf
 
* [3] The implementation of Ping Pong - Atari in tensorflow 1 of "coreystaten".

  https://github.com/coreystaten/deeprl-ppo
  
* [4] Some of parameters of the convolutional neural network of "jakegrigsby".

  https://github.com/jakegrigsby/supersonic/tree/master/supersonic

* [5] OpenAI Baselines of Atari and Retro wrappers for pre processing.

  https://github.com/openai/baselines/tree/master/baselines
  
* [6] The implementation of Super Mario Brothers by "Kautenja".

  https://github.com/Kautenja/gym-super-mario-bros
 
### What to do now?
* Implement meta learning (joint PPO) and train in multiple enviroments for a more generalized actor.
