# PPO-Mario-Bros-Tensorflow-2
A modular implementation for Proximal Policy Optimization in Tensorflow 2 using Eagerly Execution for the Super Mario Bros enviroment.

![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/SI_3DSVC_SuperMarioBros_image1600w.jpg)
## Requeirements:
- Tensorflow 2
- OpenCV
- OpenIA gym
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

Where the first argument of test is the number of episodes to test the model, and the second is the number of the enviroment to test,
for the code the enviroments of test are the next ones:
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

Eight actors were trained in the first level of Mario, and this is how it learned to finish it.

![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/mario.gif)

A plot how the average reward evolved vs the time steps, the model trained in four steps due connection, the reward isn't the same as the raw output of Kautenja's implementation, it was previously scaled for this model, all the data pre processing is in the Datapreprocessing.py file.

![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/log1.PNG)![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/log2.PNG)![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/log3.PNG)![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/log4.PNG)

In the logs directory you can find two more plots, for average X_position and Max_X_position. 

Testing in not observed enviroments:

![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/test_2.gif) ![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/test_3.gif) ![alt text](https://github.com/vcadillog/PPO-Mario-Bros-Tensorflow-2/blob/master/images/test_4.gif)

### This code was inspired from:
* [1] Proximal Policy Optimization Algorithms. 

  https://arxiv.org/pdf/1707.06347.pdf

* [2] Gotta Learn Fast: A New Benchmark for Generalization in RL.

  https://arxiv.org/pdf/1804.03720.pdf
 
* [3] The implementation in tensorflow 1 of "coreystaten".

  https://github.com/coreystaten/deeprl-ppo
  
* [4] Some of parameters of the convolutional neural network of "jakegrigsby".

  https://github.com/jakegrigsby/supersonic/tree/master/supersonic

* [5] OpenAI Baselines of Atari and Retro wrappers.

  https://github.com/openai/baselines/tree/master/baselines
  
* [6] The implementation of Super Mario Brothers by "Kautenja".

  https://github.com/Kautenja/gym-super-mario-bros
 
### To do:
* Implement meta learning and train in multiple enviroments for a more generalized actor.
