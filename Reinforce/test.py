from Reinforce import Reinforce
import gym
from PIL import Image


def test(n_episodes=5, name='CartPole_v0.pth'):
    env = gym.make('CartPole-v0')
    agent = Reinforce(env.action_space.n)
    agent.load()

    render = True
    save_gif = True

    for i_episode in range(1, n_episodes + 1):
        t=0
        state = env.reset()
        while True:
            running_reward = 0
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                env.render()
                if save_gif:
                    img = env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('./gif/{}.jpg'.format(t))
            t += 1
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()

if __name__ == '__main__':
    test()