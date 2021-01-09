import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import (Dense, InputLayer)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from utils.ReplayMemory import ReplayMemory
import gym
import numpy as np

class Actor(Model):
	def __init__(self,observation_shape,n_actions):
		super(Actor,self).__init__(name='Actor')	
		self.observation_shape = observation_shape
		self.action_space = n_actions

		self.input_layer = InputLayer(input_shape=self.observation_shape)
		self.fc1 = Dense(128,activation='relu')
		self.fc2 = Dense(64,activation='relu')
		self.fc3 = Dense(self.action_space,activation='softmax')

	@tf.function
	def call(self,inputs):
		x = self.fc1(inputs)
		x = self.fc2(x)
		action_probs = self.fc3(x)
		return action_probs

class Critic(Model):
	def __init__(self,observation_shape):
		super(Critic,self).__init__(name='Critic')

		self.observation_shape = observation_shape

		self.input_layer = InputLayer(input_shape=self.observation_shape)
		self.fc1 = Dense(32,activation='relu')
		self.fc2 = Dense(32,activation='relu')
		self.fc3 = Dense(1)

	@tf.function
	def call(self,inputs):
		x = self.input_layer(inputs)
		x = self.fc1(x)
		x = self.fc2(x)
		v = self.fc3(x)
		return v


class ActorCritic: 
	
	def __init__(self,env,lr=.001,gamma=1,critic_memory=100000,t_net_updt_intvl = 250,batch_size=32):
		self.env = env
		self.gamma = gamma
		self.observation_shape = env.observation_space.shape
		self.n_actions = env.action_space.n
		self.actor = Actor(self.observation_shape,self.n_actions)
		self.q_net = Critic(self.observation_shape)
		self.t_net = Critic(self.observation_shape)
		self.actor_optim = Adam(lr)
		self.critic_optim = Adam(lr)
		self.n_actions = env.action_space.n
		self.batch_size = batch_size
		self.replay_memory = ReplayMemory(critic_memory,self.observation_shape)
		self.t_net_updt_intvl = t_net_updt_intvl

	def get_action(self,state):
		action_probs = self.actor(state[None])
		distr = tfp.distributions.Categorical(action_probs)
		action = distr.sample()
		return int(action.numpy()[0])

	def loss(self,advantage,probs,action):
		distr = tfp.distributions.Categorical(probs)
		return -distr.log_prob(action) * advantage

	def train_critic(self):
		states,actions,rewards,next_states,dones = self.replay_memory.sample(self.batch_size)
		td_target = rewards + self.gamma * self.t_net(next_states)

		with tf.GradientTape() as tape:
			predictions = self.q_net(states)
			l = tf.math.reduce_mean(tf.square(td_target-predictions))
		gradients = tape.gradient(l,self.q_net.trainable_variables)
		self.critic_optim.apply_gradients(zip(gradients,self.q_net.trainable_variables))

	def save_weights(self):
		self.q_net.save('q_net')
		self.actor.save('actor')

	def learn(self,num_episodes=500,render=False):

		t = 0

		ep_reward = 0
		reward_list = []

		for episode in range(num_episodes):
			state = self.env.reset()
			episode_reward = 0

			while True:
				
				action = self.get_action(state)
				next_state,reward,done,_ = self.env.step(action)
				
				if render and episode > 250:
					env.render()

				self.replay_memory.store(state,action,reward,next_state,done)
				ep_reward += reward
				
				self.train_critic()

				if (t+1) % self.t_net_updt_intvl == 0:
					self.t_net.set_weights(self.q_net.get_weights())
				
				advantage = reward + self.gamma * self.t_net(next_state[None])- self.q_net(state[None])

				with tf.GradientTape() as tape:
					probs = self.actor(state[None])
					l = self.loss(advantage,probs,action)
				gradients = tape.gradient(l,self.actor.trainable_variables)
				self.actor_optim.apply_gradients(zip(gradients,self.actor.trainable_variables))

				state = next_state
				if done:
					break
					reward_list.append(reward)
					print(f"for episodes:{episode+1},mean reward:{np.mean(reward_list[-100:])}")

def main():
	env = gym.make('LunarLander-v2')
	agent = ActorCritic(env)
	agent.learn()
	env.close()

if __name__ == '__main__':
	main()

