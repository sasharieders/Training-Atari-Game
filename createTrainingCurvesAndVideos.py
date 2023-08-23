import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from gym.wrappers import TimeLimit
from gym.wrappers import AtariPreprocessing
from tf_agents.environments import suite_gym
from tensorflow import keras
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
from tensorflow import keras
import sys
import os
from tf_agents.environments import suite_gym
import matplotlib.pyplot as plt
from tf_agents.environments.wrappers import ActionRepeat
from gym.wrappers import TimeLimit
from tf_agents.environments import suite_atari
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation


def generate_training_curves(checkpoint_paths, output_dir):
    episode_lengths = []
    iterations = []

    for checkpoint_path in checkpoint_paths:
        iteration = int(checkpoint_path.split("_")[1].split("-")[0])
        average_episode_length = load_and_evaluate(checkpoint_path)

        iterations.append(iteration)
        episode_lengths.append(average_episode_length)

    # Plot the training curve
    plt.plot(iterations, episode_lengths)
    plt.xlabel("Iterations")
    plt.ylabel("Average Episode Length")
    plt.title("Training Curve")
    plt.savefig(output_dir + "/TrainingCurves.png")
    plt.close()

#we're going to use the average episode length metric 
from tf_agents.metrics import tf_metrics

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

logging_frequency = 10000  # Log metrics every 10,000 iterations

#we're going to use episode length as the metric for reward 
def load_and_evaluate(checkpoint_path):
    checkpoint = tf.train.Checkpoint(agent=agent)
    checkpoint.restore(checkpoint_path).expect_partial()

    # Reset the metrics
    for metric in train_metrics:
        metric.reset()

    average_episode_lengths = []

    while len(average_episode_lengths) < num_episodes:
        time_step = tf_env.reset()
        episode_step = 0

        while not time_step.is_last():
            action_step = agent.collect_policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            episode_step += 1

        average_episode_lengths.append(episode_step)

    return sum(average_episode_lengths) / len(average_episode_lengths)

for checkpoint_path in checkpoint_paths:
    iteration = int(checkpoint_path.split("_")[1].split("-")[0])
    average_episode_length = load_and_evaluate(checkpoint_path)

    if iteration % logging_frequency == 0:
        log_metrics(train_metrics)

    iterations.append(iteration)
    episode_lengths.append(average_episode_length)

generate_training_curves(checkpoint_paths, output_dir)

#now create video, just use code from class, and #just pick the checkpoints based on their location in the folder. 
#pick first, middle, 
#and last checkpoint for
#bad, okay, and good gameplay 
#here we can just call deployGamePlayer and get the videos for our policies that way.
max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "PongNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])


import tensorflow as tf
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.environments import suite_gym

# Load the saved policy
#policy 1, 2, 3

tf_env = suite_gym.load(environment_name)
time_step = env.reset()

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(fig, update_scene, fargs=(frames, patch),frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim
plot_animation(frames)

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")
mpl.rc('animation', html='jshtml')
frames = []

def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))
watch_driver = DynamicStepDriver(
    tf_env,
    #would iterate over the three uplaoded policies 
    policy,
    observers=[save_frames, ShowProgress(20000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()

image_path = '/home/jupyter/PONG/ThreeEx/animation'
import PIL
image_path = os.path.join("images", "rl", "PONG.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',append_images=frame_images[1:],save_all=True,duration=30,loop=0)