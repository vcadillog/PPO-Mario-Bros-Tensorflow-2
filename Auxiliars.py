import Common_constants
import tensorflow as tf
import os

max_steps = Common_constants.max_steps

def alpha_anneal(t):
    # return np.maximum(1.0 - (float(t) / float(max_steps)), 0.0).astype('float32')
    return tf.convert_to_tensor(np.maximum(1.0 - (float(t) / float(max_steps)), 0.0), dtype=tf.float32)
def indicator(x):
    if x:
        return 1
    else:
        return 0

def saver(nets, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i in range(len(nets)):
      if i == 0:
        name = 'value_net'
      if i == 1:
        name = 'policy_net'
      nets[i].save_weights(os.path.join(path, ("model_" + name )))
    print('Saved models')

def loader(nets, path):
    assert os.path.exists(path) == True
    assert len(nets) == 2

    for i in range(len(nets)):
      if i == 0:
        name = 'value_net'
      if i == 1:
        name = 'policy_net'
      nets[i].load_weights(os.path.join(path, ("model_" + name )))
      print('Load model ' + name)

def sum_writer(Writer, var,step,name):
    with Writer.as_default():        
        # other model code would go here
        tf.summary.scalar(name , var, step=step)
        Writer.flush()

def loader_test(nets, path):
    assert os.path.exists(path) == True
        
    name = 'policy_net'
    nets.load_weights(os.path.join(path, ("model_" + name )))
    print('Load model ' + name)