import sys
import random
import signal

import gym
import mgym as mgym
from mgym.envs.snake_env import SnakeEnv
import matplotlib.pyplot as plt


# Handle signal exit from keyboard
# https://stackoverflow.com/a/24426918
terminate = False
def signal_handling(signum,frame):           
    global terminate                         
    terminate = True   
signal.signal(signal.SIGINT,signal_handling) 



env = SnakeEnv()
fullobs = env.reset(2)

# create the plot
plt.ion()
plt.show()


# simulate one round of the game
while True:
    

    #a = random.choice(env.get_available_actions())
    a = [0,1]
    action = env.action_space.sample()
    print(action)

    fullobs,rewards,done,_ = env.step(action)
    
    # show the enviroment
    image = env.grid
    plt.imshow(image)
    plt.show()
    plt.draw()
    plt.pause(0.001)

    # if sim is done
    if done:
        break

    # if ctrl+c is pressed
    if terminate:              
        break

