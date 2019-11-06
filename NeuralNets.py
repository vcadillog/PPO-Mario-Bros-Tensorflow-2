import tensorflow as tf
import numpy as np
import Common_constants as CC
obs_shape = CC.obs_shape
num_actions = CC.num_actions

class value_nn(tf.keras.Model):
    def __init__(self):
        super(value_nn,self).__init__(name='Value_Net')
        
        self.conv1 = tf.keras.layers.Conv2D(filters = 32,
                                            kernel_size = (8, 8),
                                            strides = (4, 4),
                                            padding = 'same',
                                            input_shape = obs_shape,
                                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation = 'linear')
        
        self.conv2 = tf.keras.layers.Conv2D(filters = 64,
                                            kernel_size = (4, 4),
                                            strides = (2, 2),
                                            padding = 'same',
                                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation = 'linear')
        
        self.conv3 = tf.keras.layers.Conv2D(filters = 64,
                                            kernel_size = (3, 3),
                                            strides = (1, 1),
                                            padding = 'same',
                                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation = 'linear')
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512,
                                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation = 'relu')        
        self.out = tf.keras.layers.Dense(1,
                                         activation = 'linear',
                                         bias_initializer=tf.keras.initializers.Ones(),
                                         kernel_initializer=tf.keras.initializers.Orthogonal(.01))
    @tf.function
    def call(self,inputs):        
        x = self.conv1(inputs)        
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv3(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.flatten(x)        
        x = self.dense1(x)        
        return self.out(x)

class policy_nn(tf.keras.Model):
    def __init__(self):
        super(policy_nn,self).__init__(name='Policy_Net')        
        
        self.conv1 = tf.keras.layers.Conv2D(filters = 32,
                                            kernel_size = (8, 8),
                                            strides = (4, 4),
                                            padding = 'same',
                                            input_shape = obs_shape,
                                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation = 'linear')  
              
        self.conv2 = tf.keras.layers.Conv2D(filters = 64,
                                            kernel_size = (4, 4),
                                            strides = (2, 2),
                                            padding = 'same',
                                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation = 'linear')  
              
        self.conv3 = tf.keras.layers.Conv2D(filters = 64,
                                            kernel_size = (3, 3),
                                            strides = (1, 1),
                                            padding = 'same',
                                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation = 'linear')   
             
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512,
                                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Zeros(),
                                            activation = 'relu')                
        self.out = tf.keras.layers.Dense(num_actions,
                                         bias_initializer=tf.keras.initializers.Zeros(),
                                         activation = 'softmax')
    @tf.function
    def call(self,inputs):         
        x = self.conv1(inputs)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.conv3(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.flatten(x)        
        x = self.dense1(x)                
        return self.out(x)