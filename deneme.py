import make_env_
# from gym.spaces
import prng
import numpy as np
import random
import pprint
import matplotlib.pyplot as plt
import pandas as pd

# plt.style.use('seaborn')

# creates an multiagent environment which has reset, render, step
env = make_env_.make_env('traffic',benchmark=True)  # ! 2vs1 Swarm environment
# env = make_env_.make_env('simple_tag_guided_1v2')
# create interactive policies for each agent
# policies = [InteractivePolicy(env,i) for i in range(env.n)]
# print(env.observation_space)
# print(env.action_space)
# env.observation_space


# exit()
# print(policies)
# exit()


def sample_actions():
    action = [env.action_space[i].sample() for i in range(env.n)]
    action.insert(0, 0)
    # print(action)
    return action


def deterministic_actions():
    action = fixed()
    action.insert(0, 0)
    # print(action)
    return action


def deterministic_actions2():
    action = fixed2()
    action.insert(0, 0)
    # print(action)
    return action


def fixed():
    # act =  list(np.random.uniform(-1,0.5,4))
    act = [random.uniform(-0.1, 0.1), 0, 0, random.choice([1, 1, 1, 0, 0],)]
    return act


def fixed2():
    # act =  list(np.random.uniform(-1,0.5,4))
    act = [0, np.random.choice([1, 1, 1, 0, 0]), 0, random.uniform(-0.1, 0.1)]
    return act


for i_episode in range(3):  # number of simulations
    observation = env.reset()
    # print(len(observation[0]))
    rewards=[]
    for t in range(150):  # number of steps
        env.render()

        # my_action = [deterministic_actions2()]
        my_action = [deterministic_actions() if i < env.n /
                     2 else deterministic_actions2()for i in range(env.n)]

        next_state, reward, done, info = env.step(my_action)

        for agent in env.agents:
            # if(agent.isCollided):
                # print('-'*5)
                # print('collided!')
            agent.isCollided = False
            

        # print('*'*30)
        # print('next_state\n',next_state)
        # print('\nreward\n',reward)
        rewards.append(reward)
        # print('\ndone\n',done)
        # print('\ninfo\n',info)
        # print('*'*30)

        # print(len(observation))
        # print(observation[0].shape)
        # pprint.pprint(observation)
    # pprint.pprint(rewards)
    df = pd.DataFrame(rewards)
    df.to_csv('save/rewards{}.csv'.format(i_episode))
    # plt.plot(rewards)
    # plt.show()

env.close()

