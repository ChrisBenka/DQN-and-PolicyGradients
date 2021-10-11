import gym

from PolicyGradients.Algos.DDPG import DDPG
from PolicyGradients.Trainer.Trainer import Trainer

if __name__ == '__main__':
    parser = Trainer.get_args()
    parser = DDPG.get_args(parser)
    parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    args = parser.parse_args()

    env = gym.make(args.env_name)

    agent = DDPG(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.high.shape,
        action_lower_bound=env.action_space.low,
        action_upper_bound=env.action_space.high,
        actor_hidden_units=args.actor_hidden_units,
        critic_hidden_units=args.critic_hidden_units,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        noise_mu=args.noise_mu
    )
    trainer = Trainer(agent, env, vars(args))
    trainer()
