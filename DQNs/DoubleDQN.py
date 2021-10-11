from DQN import DQNAgent
import gym
import numpy as np
import tensorflow as tf


class DoubleDQNAgent(DQNAgent):
    def __init__(self, env, learning_rate=.001, replay_memory_size=100000, epsilon_min=.01, epsilon_max=.2,
                 discount_factor=.99, epsilon_decay_rate=.998, batch_size=16, update_target_network_intvl=300,
                 load_weights=False):
        super(DoubleDQNAgent, self).__init__(env,learning_rate,replay_memory_size,epsilon_min,
                                             epsilon_max,discount_factor,epsilon_decay_rate,batch_size,update_target_network_intvl,load_weights)

    def train_update(self):
        states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)
        argmax_actions = np.argmax(self.q_net(next_states),axis=1)
        next_action_values = tf.math.reduce_sum(self.target_network(next_states) * tf.one_hot(argmax_actions,self.n_actions),axis=1)
        td_target = rewards + self.discount_factor * next_action_values * (1-dones)

        with tf.GradientTape() as tape:
            predictions = tf.math.reduce_sum(self.q_net(states) * tf.one_hot(actions,self.n_actions),axis=1)
            loss = tf.math.reduce_mean(tf.square(td_target-predictions))
        variables = self.q_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = DoubleDQNAgent(env)
    agent.train(600)
    agent.make_video()