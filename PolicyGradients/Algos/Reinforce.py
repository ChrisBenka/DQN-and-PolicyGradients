import gym
import numpy as np
import tensorflow as tf
import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Policy(tf.keras.Model):
    def __init__(self, n_actions, hidden_units):
        super(Policy, self).__init__(name='Policy')
        self.fc1 = Dense(hidden_units, activation='relu')
        self.fc2 = Dense(n_actions)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        action_logits = self.fc2(x)
        return action_logits


def compute_discounted_rewards(rewards, gamma, normalize_rewards=True):
    discounted_rewards = []
    gammas = tf.stack([tf.pow(tf.cast(gamma, tf.float32), tf.cast(i, tf.float32)) for i in tf.range(tf.size(rewards))])
    for step in tf.range(tf.size(rewards)):
        discounted_reward = tf.reduce_sum(rewards * gammas) if step == 0 else tf.reduce_sum(
            rewards[:-step] * gammas[:-step])
        discounted_rewards.append(discounted_reward)
    discounted_rewards = tf.stack(discounted_rewards)
    if normalize_rewards:
        discounted_rewards = (discounted_rewards - tf.math.reduce_mean(discounted_rewards)) / \
                             tf.math.reduce_sum(tf.math.reduce_std(discounted_rewards))
    return discounted_rewards


def _compute_loss(action_probs, discounted_rewards):
    log_action_probs = tf.math.log(action_probs)
    return -tf.reduce_sum(log_action_probs * discounted_rewards)


class Reinforce:
    def __init__(self, env, lr, policy, gamma, max_episodes, max_eps_steps, normalize_rewards=True,
                 reward_threshold=195, display_progress_intvl=250):
        self.env = env
        self.lr = lr
        self.policy = policy
        self.policy_optimizer = Adam(learning_rate=lr)
        self.max_episodes = max_episodes
        self.max_eps_steps = max_eps_steps
        self.gamma = gamma
        self.normalize_rewards = normalize_rewards
        self.reward_threshold = reward_threshold
        self.display_progress_intvl = display_progress_intvl

        self.steps_taken = 0

    def env_step(self, action):
        state, reward, done, _ = self.env.step(action)
        return (
            state.astype(np.float32),
            np.array(reward, np.float32),
            np.array(done, np.int)
        )

    def tf_env_step(self, action: tf.Tensor):
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.uint8])

    def _run_episode(self):
        state = tf.constant(self.env.reset(), dtype=tf.float32)
        rewards = tf.TensorArray(tf.float32, 0, True)
        action_probs = tf.TensorArray(tf.float32, 0, True)

        state_shape = state.shape

        for step in tf.range(self.max_eps_steps):
            action, action_logits_step = self.get_action(state)
            action_probs_step = tf.nn.softmax(action_logits_step)[0, action]
            state, reward, done = self.tf_env_step(action)

            self.steps_taken += 1

            action_probs = action_probs.write(step, action_probs_step)
            rewards = rewards.write(step, reward)

            state.set_shape(state_shape)

            if tf.cast(done, tf.bool):
                break

        return action_probs.stack(), rewards.stack()

    def get_action(self, state):
        action_logits_step = self.policy(tf.expand_dims(state, 0))
        action = tf.squeeze(tf.random.categorical(logits=action_logits_step, num_samples=1))
        return action, action_logits_step

    def __call__(self, *args, **kwargs):
        running_reward = 0
        running_rewards, episode_rewards = [], []
        with tqdm.trange(1, self.max_episodes) as t:
            for episode in t:
                rewards = self.train()

                episode_reward = float(tf.reduce_sum(rewards))
                running_reward = .99 * running_reward + .01 * episode_reward

                running_rewards.append(running_reward)
                episode_rewards.append(episode_reward)

                t.set_description(f"Episode {episode}")
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                if running_reward >= self.reward_threshold:
                    print(
                        f"Solved at episode: {episode} mean, mean_reward: {running_reward}, steps_taken={self.steps_taken}")
                    break
        return running_rewards, episode_rewards

    def train(self):
        with tf.GradientTape() as tape:
            action_probs, rewards = self._run_episode()
            discounted_rewards = compute_discounted_rewards(rewards, self.gamma)
            loss = _compute_loss(action_probs, discounted_rewards)
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        return rewards

    def demo(self):
        state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = 0
        for _ in tf.range(self.max_eps_steps):
            action, _ = self.get_action(state)
            state, reward, done = self.tf_env_step(action)
            episode_reward += float(reward)
            env.render()
            if done:
                print(f"demo episode reward: {episode_reward}")
                return


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    seed = 19
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    policy = Policy(env.action_space.n, hidden_units=12)
    reinforce = Reinforce(env, lr=.0025, policy=policy, gamma=.99, max_episodes=10000, max_eps_steps=250,
                          reward_threshold=195)
    running_rewards, episode_rewards = reinforce()
    episodes = np.arange(len(running_rewards))
    plt.plot(episodes, running_rewards, label='running_reward')
    plt.plot(episodes, episode_rewards, label='episode_reward')
    plt.legend()
    plt.show()

    reinforce.demo()
