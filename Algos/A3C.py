import gym
import numpy as np
import tensorflow as tf
import threading
from time import sleep

def make_critic_network(obs_shape):
    inputs = tf.keras.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def make_policy_network(obs_shape, action_dim):
    inputs = tf.keras.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(action_dim)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def env_step(_env, action):
    obs, reward, done, _ = _env.step(action)
    return (
        obs.astype(np.float32),
        np.array(reward, np.float32),
        np.array(done, np.int)
    )

def tf_env_step(_env, action):
    return tf.numpy_function(env_step, [_env, action], [tf.float32, tf.float32, tf.uint8])


class ACNetwork:
    def __init__(self, state_size, action_size, name):
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.critic = make_critic_network(state_size)
        self.policy = make_policy_network(state_size, action_size)

    def update_weights_from_master(self):
        self.critic.set_weights(master_network.critic.get_weights())
        self.policy.set_weights(master_network.policy.get_weights())


class Worker:
    def __init__(self, _env, name, s_size, a_size, _policy_trainer, _critic_trainer, _global_episodes):
        self.name = "worker" + str(name)
        self.number = name
        self.env = _env
        self.s_size = s_size
        self.a_size = a_size
        self.global_episodes = _global_episodes
        self.policy_trainer = _policy_trainer
        self.critic_trainer = _critic_trainer

        self.episode_rewards = []
        self.episode_lengths = []
        self.running_reward = 0
        self.mean_rewards = []

        self.local_AC = ACNetwork(s_size, a_size, self.name)

    def get_action(self, state):
        action_logits_step = self.local_AC.policy(tf.expand_dims(state, 0))
        action = tf.squeeze(tf.random.categorical(logits=action_logits_step, num_samples=1))
        return action, action_logits_step

    def _run_episode(self, max_eps_steps):
        state = tf.constant(self.env.reset(), dtype=tf.float32)
        rewards = tf.TensorArray(tf.float32,0,True)
        action_probs = tf.TensorArray(tf.float32,0,True)
        values = tf.TensorArray(tf.float32,0,True)

        with tf.GradientTape() as policyTape, tf.GradientTape() as criticTape:
            for step in tf.range(max_eps_steps):
                value = self.local_AC.critic(state[None])
                action, action_logits_step = self.get_action(state)
                action_probs_step = tf.nn.softmax(action_logits_step)[0, action]
                state, reward, done = tf_env_step(action)

                values.write(step,value)
                rewards.write(step,reward)
                action_probs.write(step,action_probs_step)

                if tf.cast(done,tf.bool):
                    episode_reward = tf.reduce_sum(rewards.stack())
                    running_reward = .99 * self.running_reward + .01 * episode_reward
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(step)
                    self.mean_rewards.append(running_reward)
                    break
            rewards = rewards.stack()
            for t in




        return None, None

    def work(self, max_steps, gamma, coord):
        print(f"Starting worker {self.name}")
        while not coord.should_stop():
            self.local_AC.update_weights_from_master()

            policy_grads, critic_grads = self._run_episode(max_eps_steps=max_steps)

            policy_trainer.apply_gradients(zip(policy_grads, master_network.policy.trainable_variables))
            critic_trainer.apply_gradients(zip(critic_grads, master_network.critic.trainable_variables))

            self.global_episodes.assign_add(1)

            if self.global_episodes >= max_episodes:
                coord.request_stop()
            else:
                print(self.global_episodes)
        print(f"exiting worker {self.name}")


if __name__ == '__main__':
    max_episode_length = 250
    max_episodes = 100
    gamma = .99
    # modified per env
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    state_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)

    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global episodes', trainable=False)
        policy_trainer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        critic_trainer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        master_network = ACNetwork(state_size=state_shape, action_size=action_shape, name='global')

        num_workers = 4
        workers = []
        for i in range(num_workers):
            workers.append(
                Worker(gym.make(env_name), i, state_shape, action_shape,
                       policy_trainer, critic_trainer, global_episodes)
            )
        coord = tf.train.Coordinator()
        worker_threads = []
        for worker in workers:
            def worker_work(): worker.work(max_episode_length, gamma, coord)
            t = threading.Thread(target=worker_work)
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
