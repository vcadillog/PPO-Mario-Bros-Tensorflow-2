import Enviroments

env = Enviroments.make_env(0)
obs_shape = env.observation_space.shape
num_actions = env.action_space.n
num_actors = 8
gae_lambda = 0.95
gamma = 0.99 
base_clip_epsilon = 0.2
max_steps = 1e7
base_learning_rate = 2.5e-5
horizon = 1280
batch_size = 320
optim_epochs = 5
value_loss_coefficient = .01
entropy_loss_coefficient = .01
gradient_max = 10.0
start_t = 0

load_model = True
save_path = "./saved_models"
log_dir = "./logs"

SMALL_NUM = 1e-8

env_name = 'Dummy_Name_To_Update_To_Levels'
