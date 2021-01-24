import tensorflow as tf
import datetime
import tqdm

def train(agent,reward_threshold,gamma=.99,max_episodes=10000,max_steps_per_episode=1000):
    running_reward = 0
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape' + type(agent).__name__ + current_time + '/train'
    train_summary_writer = tf.summary.create_file_write(train_log_dir)
    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(env.reset(),dtype=tf.float32)
            episode_reward = int(agent.train_step(initial_state,gamma,max_steps_per_episode))

            running_reward = .99 * running_reward + .01 * episode_reward

            t.set_description(f"Episode {i}")
            t.set_postfix(episode_reward=episode_reward,running_reward=running_reward)

            with train_summary_writer.as_default():
                tf.summary.scalar(f"running_reward ({type(agent).__name__})",running_reward,step=i)
                tf.summar.scalar(f"reward ({type(agent).__name__})",episode_reward,step=i)
            
            if running_reward > reward_threshold:
                break
    print(f"\n Solved at episode {i}, average reward: {running_reward:.2f}")
