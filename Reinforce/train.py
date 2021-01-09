import gym
from Reinforce import Reinforce
from ReinforceWithBaseline import ReinforceWithBaseline
import numpy as np
from matplotlib import pyplot as plt

def train(agent,env):
    num_episodes = 650
    reward_list, mean_rewards = [], []
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        episode_reward = 0
        state = env.reset()
        while True:
            action = agent.get_action(state)
            next_state,reward,done,_ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            episode_reward += reward

            if done:
                agent.train_iter(states,actions,rewards)

                reward_list.append(episode_reward)
                mean_reward = np.mean(reward_list[-100:])

                if mean_reward >= 180 and mean_reward >= mean_rewards[episode-1]:
                    agent.save()
                if mean_reward >= 195:
                    print(f"SOLVED at episode:{episode}")
                mean_rewards.append(mean_reward)
                print(f"episode:{episode},reward:{episode_reward},mean_reward:{mean_reward}")
                break

    return reward_list,mean_rewards

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    lr = .009
    alpha = 1
    t = 0
    for agent in [Reinforce(env.action_space.n,lr=lr),ReinforceWithBaseline(env.action_space.n,lr=lr)]:
        reward_list,mean_rewards = train(agent,env)
        plt.plot(reward_list,label=f"{agent.__class__.__name__}-episode-reward",alpha=alpha*(.5**t))
        plt.plot(mean_rewards,label=f"{agent.__class__.__name__}-mean-reward")
        print(f"avg of last 50 episodes:{np.mean(reward_list[-50:])}")
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.legend()
    plt.title('Reinforce V ReinforceWithBaseLine')
    plt.savefig('ReinforceWithBaseline')
