import argparse
import numpy as np
import time
import pickle
import signal


# import our gym enviroments
import gym
import mgym as mgym
from mgym.envs.snake_env import SnakeEnv

# for plotting of the enviroment
# import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore


# import maddpg modules and tensorflow
import tensorflow as tf
import tensorflow.contrib.layers as layers
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer


# Handle signal exit from keyboard
# https://stackoverflow.com/a/24426918
sigterm = False
def signal_handling(signum,frame):           
    global sigterm                         
    sigterm = True   
signal.signal(signal.SIGINT,signal_handling)



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=200, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes")
    parser.add_argument("--num-agents", type=int, default=4, help="number of agents in the system")
    parser.add_argument("--agent-policy", type=str, default="maddpg", help="policy (maddpg or ddpg)")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=256, help="number of units in the mlp")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--save-dir", type=str, default="./models/", help="directory where data is saved")
    return parser.parse_args()




def mlp_model(input, num_outputs, scope, reuse=False, num_units=256, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def get_trainers(env, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(env.N):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.agent_policy=='ddpg')))
    return trainers





##============================================================================================
##============================================================================================

# read in args
arglist = parse_args()

# load with tensorflow enabled
with U.single_threaded_session():

    # Create environment
    env = SnakeEnv()
    env.reset(arglist.num_agents)

    # Create the plot for this enviroment
    #plt.show(block=False)
    plot_obj = pg.image(env.grid)
    plot_obj.setLevels(min=0,max=env.N+2)
    plot_obj.setPredefinedGradient("thermal")
    QtGui.QApplication.processEvents()

    # Create agent trainers (note that we flatten the observation space)
    obs_shape_n = [[env.observation_space.shape[0]*env.observation_space.shape[1]] for i in range(env.N)]
    trainers = get_trainers(env, obs_shape_n, arglist)
    print('Using policy \'{}\' for all agents'.format(arglist.agent_policy))

    # Load previous results, if necessary
    if arglist.restore:
        print('Loading previous state...')
        U.load_state(arglist.save_dir)

    # Initialize
    U.initialize()

    # simulation parameters
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.N)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    agent_info = [[[]]]  # placeholder for benchmarking info
    saver = tf.train.Saver()
    obs_n = env.grid.flatten()
    episode_step = 0
    train_step = 0
    t_start = time.time()


    print('\n\n')
    print('Starting iterations...')
    while not sigterm:

        # 1. get action, which is the argmax of the current policy
        action_n_full = [agent.action(obs_n) for agent in trainers]
        action_n = [np.argmax(actions) for actions in action_n_full]
        #print(action_n_full)
        #print(action_n)
        
        # 2. environment step
        new_obs_n_2d, rew_n, done_n, info_n = env.step(action_n)
        new_obs_n = new_obs_n_2d.flatten()
        episode_step += 1
        terminal = (episode_step >= arglist.max_episode_len)

        # 3. collect experience and rewards
        for i, agent in enumerate(trainers):
            agent.experience(obs_n, action_n_full[i], rew_n[i], new_obs_n, done_n, terminal)
        obs_n = new_obs_n

        # 4 record our history of rewards
        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        # 5. exit this simulation if we are done (reset the sim env)
        if done_n or terminal:
            env.reset(arglist.num_agents)
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # 6. increment global step counter
        train_step += 1

        # 7. for displaying learned policies (once we start training)
        if arglist.display:
            plot_obj.setImage(env.grid)
            plot_obj.setLevels(min=0,max=env.N+2)
            QtGui.QApplication.processEvents()
            

        # 8. update all trainers, if not in display or benchmark mode
        losses = []
        for agent in trainers:
            agent.preupdate()
        for agent in trainers:
            loss = agent.update(trainers, train_step)
            if loss:
                losses.append(loss[0:2])


        # 9. display the current results!
        if not len(losses) == 0:
            print("agents loss: {}".format([[round(y,3) for y in x] for x in losses]))
        if done_n or terminal:
            print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                train_step, len(episode_rewards), int(np.mean(episode_rewards[-arglist.max_episode_len:])),
                    [int(np.mean(rew[-arglist.max_episode_len:])) for rew in agent_rewards], round(time.time()-t_start, 3)))


        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > arglist.num_episodes:
            U.save_state(arglist.save_dir, saver=saver)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break










