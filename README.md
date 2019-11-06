# PPO-Mario-Bros-Tensorflow-2
A modular implementation for Proximal Policy Optimization in Tensorflow 2 using Eagerly Execution for the Super Mario Bros enviroment.

Requeirements:
- Tensorflow 2
- OpenCV
- OpenIA gym
- Super Mario Bros NES, developed by Kautenja

How to use:
Clone the repository,

For training run:
!python -c 'from Main import train; test(True)'
The argument of test enable the load of the trained model.

For testing the model:
!python -c 'from Main import test; test(10,0)'
Where the first argument of test is the number of episodes to test the model, and the second is the number of the enviroment to test,
for the code the enviroments of test are the next ones:
0 : SuperMarioBros-1-1-v0
The first level of the first world
1 : SuperMarioBros-1-2-v0 
The second level of the first world
2 : SuperMarioBros-1-3-v0
The third level of the first world
3 : SuperMarioBros-2-2-v0
The second level of the second world
