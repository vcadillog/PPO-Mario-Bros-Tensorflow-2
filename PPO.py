import tensorflow as tf
import Common_constants as CC

num_actions = CC.num_actions
base_clip_epsilon = CC.base_clip_epsilon
value_loss_coefficient = CC.value_loss_coefficient
entropy_loss_coefficient = CC.entropy_loss_coefficient

@tf.function(experimental_relax_shapes=True)
def loss(alpha,policy_network,value_network,observation,actions,advantages,old_policies,old_values):    
    policies = policy_network(observation)    
    values = value_network(observation)
    values = tf.squeeze(values, axis = 1)    
    act_onehot = tf.one_hot(actions, num_actions)    
    clip_epsilon = tf.math.multiply(alpha , base_clip_epsilon)
    log_prob_ratio = tf.math.log(tf.reduce_sum(policies * act_onehot, axis=1)) - tf.math.log(tf.reduce_sum(old_policies * act_onehot, axis=1))                                   
    prob_ratio = tf.math.exp(log_prob_ratio)             
    clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon)                                            
    entropy_loss = -tf.reduce_sum(- policies * tf.math.log(policies), axis=1)                        
    clip_loss = -tf.math.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages)                                         
    value_loss = tf.math.square(values - old_values)        
    total_loss = tf.reduce_mean(clip_loss + value_loss_coefficient * value_loss + entropy_loss_coefficient * entropy_loss)            
    return entropy_loss, clip_loss, value_loss, total_loss


def gradients(optimizer):  
    @tf.function
    def apply_grads(alpha,policy_network,value_network,observation,actions,advantages,old_policies,old_values):
        
        with tf.GradientTape() as tape:                  
            entropy_loss, clip_loss, value_loss, total_loss = ppo_loss(alpha,policy_network,value_network,observation,actions,advantages,old_policies,old_values)
        variables = (policy_network.trainable_variables + value_network.trainable_variables)
        gradients = tape.gradient(total_loss, variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, gradient_max)
        optimizer.apply_gradients(zip(clipped_gradients, variables))     
        return entropy_loss, clip_loss, value_loss, total_loss
    return apply_grads