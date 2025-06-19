from torch.utils.data import Dataset
import torch
import numpy as np
from minari.dataset.minari_dataset import MinariDataset


class MORelDataset(Dataset):

    def __init__(self, dataset: MinariDataset):

        source_observation_list = []
        source_action_list = []
        target_delta_list = []
        target_reward_list = []

        episodes_generator = dataset.iterate_episodes()
        for episode in episodes_generator:
            # Input data
            source_observation = episode.observations[:-1]
            source_action = episode.actions

            # Output data
            target_delta = episode.observations[1:] - source_observation
            target_reward = episode.rewards

            source_observation_list.append(source_observation)
            source_action_list.append(source_action)
            target_delta_list.append(target_delta)
            target_reward_list.append(target_reward)

        self.source_observation = np.concatenate(source_observation_list, axis=0)
        self.source_action = np.concatenate(source_action_list, axis=0)
        self.target_delta = np.concatenate(target_delta_list, axis=0)
        self.target_reward = np.concatenate(target_reward_list, axis=0)
        # Normalize data
        self.delta_mean = self.target_delta.mean(axis=0)
        self.delta_std = self.target_delta.std(axis=0) + 1e-8

        self.reward_mean = self.target_reward.mean(axis=0)
        self.reward_std = self.target_reward.std(axis=0)

        self.observation_mean = self.source_observation.mean(axis=0)
        self.observation_std = self.source_observation.std(axis=0)

        self.action_mean = self.source_action.mean(axis=0)
        self.action_std = self.source_action.std(axis=0)

        self.source_action = (self.source_action - self.action_mean) / self.action_std
        self.source_observation = (
            self.source_observation - self.observation_mean
        ) / self.observation_std
        self.target_delta = (self.target_delta - self.delta_mean) / self.delta_std
        self.target_reward = (self.target_reward - self.reward_mean) / self.reward_std

    def __getitem__(self, idx):
        feed = torch.FloatTensor(
            np.concatenate([self.source_observation[idx], self.source_action[idx]])
        )
        target = torch.FloatTensor(
            np.append(self.target_delta[idx], self.target_reward[idx])
        )

        return feed, target

    def __len__(self):
        return len(self.source_observation)
