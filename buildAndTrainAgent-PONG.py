import tensorflow as tf
import gym
from gym.wrappers import TimeLimit
from gym.wrappers import AtariPreprocessing
from tf_agents.environments import suite_gym
from tensorflow import keras
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
max_episode_steps = 27000
environment_name = "PongNoFrameskip-v4"
env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])
from tf_agents.environments.tf_py_environment import TFPyEnvironment
tf_env = TFPyEnvironment(env)
from tf_agents.networks.q_network import QNetwork
import numpy as np
preprocessing_layer = tf.keras.layers.Lambda(
    lambda obs: tf.cast(obs, np.float32) / 255.
)
conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params = [512]

q_net = QNetwork(
    tf_env.time_step_spec().observation,
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params
)
from tf_agents.agents.dqn.dqn_agent import DqnAgent
# see TF-agents issue #113
optimizer = tf.keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
epsilon=0.00001, centered=True)
train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
#optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95,
# momentum=0.0,epsilon=0.00001, centered=True)
epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ?
    decay_steps=25000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ?
agent = DqnAgent(tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=20000, # <=> 32,000 ALE frames
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
                 #changed from .95 to .8
        gamma=0.99, # discount factor
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()
from tf_agents.replay_buffers import tf_uniform_replay_buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000)
replay_buffer_observer = replay_buffer.add_batch
from tf_agents.metrics import tf_metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]
#log these
from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period)
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 10000 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")
from tf_agents.policies.random_tf_policy import RandomTFPolicy
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000)
final_time_step, final_policy_state = init_driver.run()
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)
from tf_agents.utils.common import function
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)
checkpoint = tf.train.Checkpoint(agent)
import os
import tensorflow as tf
from tf_agents.policies.policy_saver import PolicySaver
def train_agent(n_iterations, checkpoint_interval, checkpoint_dir):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    trajectories_list = []

    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        trajectories_list.append(trajectories)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(iteration, train_loss.loss.numpy()), end="")

        if iteration % checkpoint_interval == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
            checkpoint = tf.train.Checkpoint(agent=agent)
            checkpoint.save(checkpoint_path)

    return trajectories_list

n_iterations = 1000000
checkpoint_interval = 10000


checkpoint_dir = "/home/jupyter/PONG/checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

trajectories = train_agent(n_iterations, checkpoint_interval, checkpoint_dir)