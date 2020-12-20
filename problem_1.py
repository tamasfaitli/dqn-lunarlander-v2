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
PAR_N_MAX_EPISODES      = 600
PAR_DISCOUNT_FACTOR     = 0.99
PAR_EXP_BUFFER_SIZE     = 15000
PAR_BATCH_SIZE          = 64
# PAR_TARGET_UPDATE_FREQ  = int(PAR_EXP_BUFFER_SIZE/PAR_BATCH_SIZE)
PAR_TARGET_UPDATE_FREQ  = 100
PAR_LEARNING_RATE       = 0.001
PAR_EPS_MAX             = 0.99
PAR_EPS_MIN             = 0.05
# PAR_EPS_Z               = int(0.91*PAR_N_MAX_EPISODES)
PAR_EPS_Z               = 350

# definitions (these are not need to be changed during tuning for this assignment)
DEF_ENV                 = 'LunarLander-v2'
DEF_OUTP_FILE           = 'neural-network-1'
DEF_N_EP_RUNNING_AVG    = 50
DEF_REWARD_THRESHOLD    = 50.0


# function definitions
def init_environment():
    '''

    :return:
    '''
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
    '''

    :param env:
    :param agent:
    :return:
    '''
    ep_rewards = []
    ep_num_of_steps = []
    ep_avg_losses = []
    episodes = trange(PAR_N_MAX_EPISODES, desc='Episode: ', leave=True)

    ep = 0
    avg_rew = 0.0

    # filling replay buffer
    state = env.reset()
    while True:
        if agent.experience_buffer.is_full():
            break

        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        # adding experience to the buffer
        agent.add_experience(state, action, reward, next_state, done)

        if done:
            env.close()
            state = env.reset()

    # train till either performance is acceptable or all episodes expire
    while (avg_rew <= DEF_REWARD_THRESHOLD) and (ep < PAR_N_MAX_EPISODES):

        # epsilon decay for epsilon-greedy policy
        eps_k = max(PAR_EPS_MIN, PAR_EPS_MAX-((PAR_EPS_MAX-PAR_EPS_MIN)*(ep)/(PAR_EPS_Z-1)))

        # init new episode
        done = False
        state = env.reset()
        ep_total_reward = 0.0
        ep_total_loss = 0.0
        t = 0
        while not done:
            if SWT_RENDER[0] and (ep%SWT_RENDER[1]==0):
                env.render()

            # evaluate epsilon-greedy policy
            action = agent.action(state, eps_k)

            # step the environment
            next_state, reward, done, _ = env.step(action)

            # adding experience to the buffer
            l = agent.add_experience(state, action, reward, next_state, done)

            # update episode measures
            ep_total_reward += reward
            ep_total_loss += l

            # prepare for next episode
            state = next_state
            t += 1

        # save episode measures
        ep_rewards.append(ep_total_reward)
        ep_num_of_steps.append(t)
        ep_avg_losses.append(ep_total_loss/t)

        env.close()

        avg_rew = running_average(ep_rewards, DEF_N_EP_RUNNING_AVG)[-1]
        ep += 1

        episodes.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Eps: {} - Avg. Reward/Steps: {:.1f}/{}".format(
                ep, ep_total_reward, t, eps_k,
                avg_rew,
                running_average(ep_num_of_steps, DEF_N_EP_RUNNING_AVG)[-1]))

    return ep_rewards, ep_num_of_steps, ep_avg_losses


def plot_results(rewards, steps, losses):
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

    plt.figure()
    plt.plot(losses)

    plt.show()


def save_network(agent):
    '''

    :param agent: Agent containing the DQN network
    :return:
    '''
    torch.save(agent.network, DEF_OUTP_FILE+".pth")

# main
if __name__ == '__main__':
    # initializing environment
    env = init_environment()
    # initializing agent
    agent = Agent(env, PAR_DISCOUNT_FACTOR, PAR_EXP_BUFFER_SIZE, PAR_BATCH_SIZE, PAR_TARGET_UPDATE_FREQ, PAR_LEARNING_RATE)
    # train the agent
    rewards, steps, losses = training(env, agent)
    # save network file
    save_network(agent)
    # plot rewards, steps, and loss
    plot_results(rewards, steps, losses)
