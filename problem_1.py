#################################################
#                                               #
# EL2805 Reinforcement Learning                 #
# Computer Lab 2                                #
# Problem 1                                     #
#                                               #
# Author: Tamas Faitli (19960205-T410)          #
#                                               #
#################################################

# imports
import numpy as np
import matplotlib.pyplot as plt
import gym
from agent import Agent
from tqdm import trange
import torch

# switches
SWT_RENDER              = [False, 100] # [render or not, if yes each N-th episode]

# parameters
PAR_N_MAX_EPISODES      = 900
PAR_DISCOUNT_FACTOR     = 0.995
PAR_EXP_BUFFER_SIZE     = 15000
PAR_BATCH_SIZE          = 80
PAR_TARGET_UPDATE_FREQ  = int(PAR_EXP_BUFFER_SIZE/PAR_BATCH_SIZE)
PAR_LEARNING_RATE       = 0.0005
PAR_EPS_MAX             = 0.99
PAR_EPS_MIN             = 0.05
PAR_EPS_Z               = int(0.925*PAR_N_MAX_EPISODES)

# definitions (these are not need to be changed during tuning for this assignment)
DEF_ENV                 = 'LunarLander-v2'
DEF_OUTP_FILE           = 'neural-network-1'
DEF_N_EP_RUNNING_AVG    = 50
DEF_REWARD_THRESHOLD    = 50.0


# function definitions
def init_environment():
    env = gym.make(DEF_ENV)
    env.reset()
    return env

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def training(env, agent):
    ep_rewards = []
    ep_num_of_steps = []
    episodes = trange(PAR_N_MAX_EPISODES, desc='Episode: ', leave=True)

    ep = 0
    avg_rew = 0.0

    while not agent.experience_buffer.is_full():
        state = env.reset()
        done = False
        while not done:
            action = agent.action(state, 1)

            next_state, reward, done, _ = env.step(action)

            # adding experience to the buffer
            agent.add_experience(state,action,reward,next_state,done)
        env.close()



    while (avg_rew <= DEF_REWARD_THRESHOLD) and (ep < PAR_N_MAX_EPISODES):

        eps_k = max(PAR_EPS_MIN, PAR_EPS_MAX-((PAR_EPS_MAX-PAR_EPS_MIN)*(ep)/(PAR_EPS_Z-1)))

        # new episode
        done = False
        state = env.reset()
        ep_total_reward = 0.0
        t = 0
        while not done:
            if SWT_RENDER[0] and (ep%SWT_RENDER[1]==0):
                env.render()


            action = agent.action(state, eps_k)

            next_state, reward, done, _ = env.step(action)

            # adding experience to the buffer
            agent.add_experience(state, action, reward, next_state, done)

            ep_total_reward += reward

            state = next_state
            t += 1

        ep_rewards.append(ep_total_reward)
        ep_num_of_steps.append(t)

        env.close()

        avg_rew = running_average(ep_rewards, DEF_N_EP_RUNNING_AVG)[-1]
        ep += 1

        episodes.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                ep, ep_total_reward, t,
                avg_rew,
                running_average(ep_num_of_steps, DEF_N_EP_RUNNING_AVG)[-1]))

    return ep_rewards, ep_num_of_steps


def plot_results(rewards, steps):
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, len(rewards) + 1)], rewards, label='Episode reward')
    ax[0].plot([i for i in range(1, len(rewards) + 1)], running_average(
        rewards, DEF_N_EP_RUNNING_AVG), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, len(steps) + 1)], steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, len(steps) + 1)], running_average(
        steps, DEF_N_EP_RUNNING_AVG), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()


def save_network(agent):
    torch.save(agent.network, DEF_OUTP_FILE+".pth")


# main
if __name__ == '__main__':
    env = init_environment()
    agent = Agent(env, PAR_DISCOUNT_FACTOR, PAR_EXP_BUFFER_SIZE, PAR_BATCH_SIZE, PAR_TARGET_UPDATE_FREQ, PAR_LEARNING_RATE)

    rewards, steps = training(env, agent)
    save_network(agent)
    plot_results(rewards, steps)
