import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from models.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from models.replay_memory import ReplayMemory
from env.dynamics_env import FakeEnv
import minari

from data.dataset import MORelDataset
from models.Dynamics import DynamicsEnsemble
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
parser.add_argument(
    "--env-name",
    default="Reacher-v5",
    help="Mujoco Gym environment (default: Reacher-v5)",
)
parser.add_argument(
    "--policy",
    default="Gaussian",
    help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
)
parser.add_argument(
    "--eval",
    type=bool,
    default=True,
    help="Evaluates a policy a policy every 10 episode (default: True)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor for reward (default: 0.99)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    metavar="G",
    help="target smoothing coefficient(τ) (default: 0.005)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0003,
    metavar="G",
    help="learning rate (default: 0.0003)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.2,
    metavar="G",
    help="Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)",
)
parser.add_argument(
    "--automatic_entropy_tuning",
    type=bool,
    default=False,
    metavar="G",
    help="Automaically adjust α (default: False)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=123456,
    metavar="N",
    help="random seed (default: 123456)",
)
parser.add_argument(
    "--batch_size", type=int, default=256, metavar="N", help="batch size (default: 256)"
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1000000,
    metavar="N",
    help="maximum number of steps (default: 1000000)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    metavar="N",
    help="hidden size (default: 256)",
)
parser.add_argument(
    "--updates_per_step",
    type=int,
    default=1,
    metavar="N",
    help="model updates per simulator step (default: 1)",
)
parser.add_argument(
    "--start_steps",
    type=int,
    default=10000,
    metavar="N",
    help="Steps sampling random actions (default: 10000)",
)
parser.add_argument(
    "--target_update_interval",
    type=int,
    default=1,
    metavar="N",
    help="Value target update per no. of updates per step (default: 1)",
)
parser.add_argument(
    "--replay_size",
    type=int,
    default=1000000,
    metavar="N",
    help="size of replay buffer (default: 10000000)",
)
parser.add_argument("--cuda", action="store_true", help="run on CUDA (default: False)")
args = parser.parse_args()

# Tesnorboard
writer = SummaryWriter(
    "runs/{}_SAC_{}_{}_{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.env_name,
        args.policy,
        "autotune" if args.automatic_entropy_tuning else "",
    )
)

# Train a dynamics model by MOReL -> env: fake_env
dataset = minari.load_dataset("mujoco/reacher/expert-v0", download=True)
dataset.set_seed(seed=args.seed)

dynamics_model = DynamicsEnsemble(
    input_dim=dataset.observation_space.shape[0] + dataset.action_space.shape[0],
    output_dim=dataset.observation_space.shape[0] + 1,
    n_models=4,
    n_neurons=args.hidden_size,
    threshold=1.5,
    n_layers=2,
    activation=torch.nn.ReLU,
    cuda=args.cuda,
)

dataset = MORelDataset(dataset=dataset)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

dynamics_model.train(dataloader=dataloader, epochs=5, summary_writer=writer)

# fake environment to augment the replay buffer
fake_env = FakeEnv(
    dynamics_model,
    dataset.observation_mean,
    dataset.observation_std,
    dataset.action_mean,
    dataset.action_std,
    dataset.delta_mean,
    dataset.delta_std,
    dataset.reward_mean,
    dataset.reward_std,
    timeout_steps=300,
    uncertain_penalty=-100,
    device="cpu",
)
fake_env.action_space.seed(args.seed)

# true environment
true_env = gym.make(args.env_name)
true_env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Agent
agent = SAC(true_env.observation_space.shape[0], true_env.action_space, args)


# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    truncated = False
    state, _ = fake_env.reset()

    while not done and not truncated:
        if args.start_steps > total_numsteps:
            action = torch.Tensor(
                fake_env.action_space.sample()
            )  # Sample random action
        else:
            action = torch.Tensor(
                agent.select_action(state)
            )  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = (
                    agent.update_parameters(memory, args.batch_size, updates)
                )

                writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)
                writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                writer.add_scalar("entropy_temprature/alpha", alpha, updates)
                updates += 1

        next_state, reward, done, truncated, info = fake_env.step(action, state)  # Step
        if not info["HALT"]:
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = (
                1.0 if episode_steps == fake_env._max_episode_steps else float(not done)
            )

            memory.push(
                state, action, reward, next_state, mask
            )  # Append transition to memory

            state = next_state
        else:
            continue
    if total_numsteps > args.num_steps:
        break

    writer.add_scalar("reward/train", episode_reward, i_episode)
    print(
        "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
            i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
        )
    )

    # Evaluate
    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.0
        episodes = 10
        for _ in range(episodes):
            state, _ = true_env.reset()
            episode_reward = 0
            done = False
            truncated = False
            while not done and not truncated:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, truncated, info = true_env.step(action)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar("avg_reward/test", avg_reward, i_episode)

        print("----------------------------------------")
        print(
            "Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2))
        )
        print("----------------------------------------")

true_env.close()
fake_env.close()
