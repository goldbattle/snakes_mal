import argparse
import numpy as np
import tensorflow as tf
import time
import sys
import pickle

import gym.spaces as spaces
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers




##============================================================================================
##============================================================================================
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out
##============================================================================================
##============================================================================================



# initialize tensorflow 
with U.single_threaded_session():
    U.initialize()


    # our trainers, the nn model, and the space size
    num_agents = 5
    trainers = []
    model = mlp_model
    observation_space = [spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32) for i in range(num_agents)]
    obs_shape_n = [observation_space[i].shape for i in range(num_agents)]
    action_space = [spaces.Discrete(5) for i in range(num_agents)]
    

    # insert a few agents in the enviroment
    arglist = parse_args()
    for i in range(num_agents):
            trainers.append(MADDPGAgentTrainer("agent_%d" % i, model, obs_shape_n, action_space, i, arglist, local_q_func=True))

    # do a simple update
    train_step = 0
    for agent in trainers:
        agent.preupdate()
        agent.action
    for agent in trainers:
        loss = agent.update(trainers, train_step)


    # debug print?????
    obs_n = [observation_space[i].sample() for i in range(num_agents)]
    action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
    print(action_n)
    for i in range(num_agents):
        print("agent "+str(i)+" took action "+str(action_n[i]))






