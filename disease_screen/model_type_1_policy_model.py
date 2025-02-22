from typing import Callable
from gymnasium import spaces
from torch import nn, reshape, ones
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    Distribution,
)
from stable_baselines3.common.type_aliases import PyTorchObs
from typing import Optional, Tuple
from disease_screen_logger import logger


class ACNet(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch
    ):
        super().__init__()
        # Policy network
        # 此处的policy net和value net输出的都是中间值，最后还要跟一个线性层，因此最后都做非线性输出
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, net_arch['pi'][0]),
            nn.ReLU(),
            nn.Linear(net_arch['pi'][0], net_arch['pi'][1]),
            nn.ReLU(),
            nn.Linear(net_arch['pi'][1], net_arch['pi'][2]),
            nn.ReLU(),
        )
        self.latent_dim_pi = net_arch['pi'][2]

        value_net_length = len(net_arch['vf'])
        # Value network
        if value_net_length == 1:
            self.value_net = nn.Sequential(
                nn.Linear(feature_dim, net_arch['vf'][0]),
                nn.ReLU(),
            )
            logger.info(f'value net length is 1')
            self.latent_dim_vf = net_arch['vf'][0]
        elif value_net_length == 2:
            self.value_net = nn.Sequential(
                nn.Linear(feature_dim, net_arch['vf'][0]),
                nn.ReLU(),
                nn.Linear(net_arch['vf'][0], net_arch['vf'][1]),
                nn.ReLU(),
            )
            logger.info(f'value net length is 2')
            self.latent_dim_vf = net_arch['vf'][1]
        else:
            assert value_net_length == 3
            self.value_net = nn.Sequential(
                nn.Linear(feature_dim, net_arch['vf'][0]),
                nn.ReLU(),
                nn.Linear(net_arch['vf'][0], net_arch['vf'][1]),
                nn.ReLU(),
                nn.Linear(net_arch['vf'][1], net_arch['vf'][2]),
                nn.ReLU(),
            )
            logger.info(f'value net length is 3')
            self.latent_dim_vf = net_arch['vf'][2]

        self.policy_net.apply(ACNet.initialize_weights)
        self.value_net.apply(ACNet.initialize_weights)

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization for linear layers
            nn.init.zeros_(m.bias)  # Initialize biases to zero

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class SymptomInquireActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        symptom_index_dict,
        action_mapping_dict,
        symptom_num,
        mask_history_action,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        self.symptom_idx_dict = symptom_index_dict
        self.action_mapping_dict = action_mapping_dict
        self.symptom_num = symptom_num
        self.features_dim = observation_space.shape[0]
        self.net_arch = kwargs['net_arch']
        self.mask_history_action = mask_history_action
        self.index_mask_dict = self.masking_set()

    def masking_set(self):
        idx_dict = self.action_mapping_dict
        index_mask_dict = dict()
        for action_index in idx_dict:
            parent_symptom_index_set = idx_dict[action_index][2]
            for parent_symptom_idx in parent_symptom_index_set:
                if parent_symptom_idx is not None and parent_symptom_idx not in index_mask_dict and parent_symptom_idx >= 0:
                    index_mask_dict[parent_symptom_idx] = []
                if parent_symptom_idx is not None and parent_symptom_idx >= 0:
                    index_mask_dict[parent_symptom_idx].append(action_index)
        return index_mask_dict

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ACNet(
            self.features_dim,
            self.net_arch
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent_with_mask(latent_pi, features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, * self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _get_action_dist_from_latent_with_mask(self, latent_pi: th.Tensor, features: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        mask = self.get_current_masking(features)
        mask_logit = th.zeros_like(mean_actions).masked_fill_(~mask, float('-inf'))
        mean_actions = mean_actions + mask_logit
        if isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent_with_mask(latent_pi, features)

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent_with_mask(latent_pi, features)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_current_masking(self, features):
        device = self.device
        btz_size = len(features)
        feature_symptom = features[:, :self.symptom_num*3]
        feature_symptom = reshape(feature_symptom, (btz_size, self.symptom_num, 3))
        feature_positive = feature_symptom[:, :, 2] == 1

        # mask应当对齐action的数量，一开始应该是所有的都可以问的。然后mask已经问过的。True是可以问
        # action空间的前一半是和feature 一一对应的，只要feature的值是已知的就mask掉
        mask = ones([btz_size, len(self.action_mapping_dict) + 1]).to(device)
        feature_unknown = feature_symptom[:, :, 0] == 1
        feature_num = feature_unknown.shape[1]
        mask[:, :feature_num] = feature_unknown
        if self.mask_history_action:
            previous_actions = features[:, -(len(self.action_mapping_dict) + 1):]
            mask = mask - previous_actions

        for symptom_index in self.index_mask_dict:
            action_trigger_list = self.index_mask_dict[symptom_index]
            # 这里必须这么写，因为只有positive可以问，unknown和no都不能问，如果只用Unknown没办法体现no的禁止性
            parent_negative = th.unsqueeze(feature_positive[:, symptom_index], dim=1)
            # 这里必须取乘法，避免将已经问过的sub action二次归正
            mask[:, action_trigger_list] = parent_negative * mask[:, action_trigger_list]
            # mask_obs = mask.detach().to('cpu').numpy()
        mask_bool = mask == 1
        return mask_bool
