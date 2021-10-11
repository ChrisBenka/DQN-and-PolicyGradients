import tensorflow as tf
from tensorflow.keras.layers import (Dense, InputLayer)
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from gym import wrappers
import os
import numpy as np
import gym


class ReplayMemory:

    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.num_in_memory = 0
        self.n_idx = 0
        self.states = np.empty((capacity,) + state_shape, dtype=np.float32)
        self.actions = np.empty((capacity,))
        self.rewards = np.empty((capacity,))
        self.next_states = np.empty((capacity,) + state_shape, dtype=np.float32)
        self.dones = np.empty((capacity,))

    def store(self, state, action, reward, next_state, done):
        self.states[self.n_idx] = state
        self.actions[self.n_idx] = action
        self.rewards[self.n_idx] = reward
        self.next_states[self.n_idx] = next_state
        self.dones[self.n_idx] = done
        self.n_idx = (self.n_idx + 1) % self.capacity
        self.num_in_memory = min(self.num_in_memory + 1, self.capacity)

    def sample(self, batch_size):
        size = batch_size
        if size > self.num_in_memory:
            size = self.num_in_memory
        idxs = np.random.choice(self.num_in_memory, size)

        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]


class QNetwork(tf.keras.Model):

    def __init__(self, input_shape, n_actions):
        super(QNetwork, self).__init__(name='QNetwork')
        self.input_layer = InputLayer(input_shape=input_shape)
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.fc3 = Dense(n_actions)

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, env, learning_rate=.001, replay_memory_size=100000, epsilon_min=.01, epsilon_max=.2,
                 discount_factor=.99, epsilon_decay_rate=.998, batch_size=16, update_target_network_intvl=300,
                 load_weights=False):

        self.env = env
        self.n_actions = env.action_space.n
        self.state_shape = env.observation_space.shape
        self.q_net = QNetwork(input_shape=self.state_shape, n_actions=self.n_actions)
        self.target_network = QNetwork(input_shape=self.state_shape, n_actions=self.n_actions)
        self.replay_memory_size = replay_memory_size
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.discount_factor = discount_factor
        self.replay_memory = ReplayMemory(capacity=replay_memory_size, state_shape=self.state_shape)
        self.epsilon_decay_rate = epsilon_decay_rate
        self.batch_size = batch_size
        self.update_target_network_intvl = update_target_network_intvl
        self.optimizer = Adam(learning_rate)

        if load_weights:
            self.q_net = tf.keras.models.load_model("q_net")

    def save_weights(self):
        self.q_net.save('q_net')

    def train(self, episodes, can_stop=True):
        t = 0
        reward_list = []
        for episode in range(episodes):
            state = self.env.reset()
            ep_reward = 0
            while True:
                greedy_action = np.argmax(self.q_net(state[None], training=True))
                action = self.get_action(greedy_action, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_memory.store(state, action, reward, next_state, done)
                self.train_update()
                ep_reward += reward

                if (t + 1) % self.update_target_network_intvl == 0:
                    self.target_network.set_weights(self.q_net.get_weights())
                t += 1
                if done:
                    reward_list.append(ep_reward)
                    break
                else:
                    state = next_state
            self.decay_epsilon()
            if (episode + 1) > 100 and np.mean(reward_list[-100:]) >= 250 and can_stop is True:
                print(f"SOLVED at episode:{episode + 1}")
                return
            elif len(reward_list) >= 100:
                print(f"for episodes:{episode + 1},mean reward:{np.mean(reward_list[-100:])}")

    def train_update(self):
        states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)

        # target is r + max_a Q(s',a) - Q(s,a)
        # pred is Q(s,a)
        # td error is target - pred
        max_action_values = np.max(self.target_network(next_states), axis=1)
        td_target = rewards + self.discount_factor * max_action_values * (1 - dones)

        with tf.GradientTape() as tape:
            predictions = tf.math.reduce_sum(self.q_net(states) * tf.one_hot(actions, self.n_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(td_target - predictions))
        variables = self.q_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def decay_epsilon(self):

        decayed_epsilon = self.epsilon * self.epsilon_decay_rate
        self.epsilon = max(self.epsilon_min, decayed_epsilon)

    def get_action(self, best_action, epsilon):

        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return best_action

    def make_video(self, env, episodes=100):
        env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
        mean_reward = Mean()
        for episode in range(episodes):
            episode_reward = 0
            state = env.reset()
            while True:
                action = np.argmax(self.q_net(state[None], training=False))
                next_state, reward, done, _ = env.step(action)
                env.render()
                episode_reward += reward
                if done:
                    print(f"episode: {episode}, episode_reward: {episode_reward}")
                    break
                else:
                    state = next_state
        print(f"mean reward: {mean_reward.result()}")


def main():
    try:
        env = gym.make('LunarLander-v2')
        agent = DQNAgent(env, load_weights=False)
        agent.train(600)
        agent.make_video(env)
        env.close()
    except KeyboardInterrupt:
        agent.save_weights()
    finally:
        agent.save_weights()


if __name__ == '__main__':
    main()