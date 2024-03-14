__author__ = "Georgi Tancev, PhD"
__copyright__ = "© Georgi Tancev"

import torch as th
import torch.nn as nn
from torch.distributions import ContinuousBernoulli, Normal
from helper import actor_init, critic_init


def MLP(sizes,
        activation,
        output_activation=nn.Identity):
    """
    The basic multilayer perceptron architecture used.

    Parameters
    ----------
    sizes: List
        List of feature sizes, i.e.,
            [indput_dim, hidden_layer_1, ..., hidden_layer_n_dim, output_dim]

    activation: nn.Module
        Activation function for the hidden layers.

    output_activation: nn.Module
        Activation function for the output layer


    Returns
    -------
    MLP: nn.Module

    """

    layers = []
    for i in range(len(sizes)-1):
        size_in, size_out = sizes[i], sizes[i+1]
        layers.append(nn.Linear(size_in, size_out))
        layers.append(activation())
    layers.pop()
    layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """A class for the policy network."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = MLP([obs_dim] + list(hidden_sizes) + [act_dim],
                              activation)
        self.logits_net.apply(actor_init)

    def _distribution(self, obs):
        """
        Takes the observation and outputs a distribution over actions.

        Parameters
        ----------
        obs: torch.Tensor of shape (n, obs_dim)
            State observation.

        Returns
        -------
        pi: torch.distributions.Distribution
            n action distributions for each state/obs.

        """
        logits = self.logits_net(th.Tensor(obs))
        pi = ContinuousBernoulli(logits=logits)
        # pi = Normal(logits, scale=3.0e-1)
        return pi

    def _log_prob_from_distribution(self, pi, act):
        """
        Take a distribution and action, then gives the log-probability
        of the action under that distribution.

        Parameters
        ----------
        pi: torch.distributions.Distribution
            n action distributions.

        act: torch.Tensor of shape (n, act_dim)
            n action for which log likelihood is calculated.

        Returns
        -------
        log_prob: torch.Tensor of shape (n, )
            log likelihood of act.

        """
        log_p = pi.log_prob(act)
        return log_p.sum(dim=-1)

    def forward(self, obs, act=None):
        """
        Produce action distributions for given observations, and then compute
        the log-likelihood of given actions under those distributions.

        Parameters
        ----------
        obs: torch.Tensor of shape (n, obs_dim)
            State observation.
        act: (torch.Tensor of shape (n, act_dim), Optional). Defaults to None.
            Action for which log likelihood is calculated.

        Returns
        -------
        pi: torch.distributions.Distribution
            n action distributions.
        log_prob: torch.Tensor of shape (n, )
            log likelihood of act.
        """
        pi = self._distribution(obs)
        log_prob = None
        if act is not None:
            log_prob = self._log_prob_from_distribution(pi, act)
        return pi, log_prob


class Critic(nn.Module):
    """The network used by the value function."""
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = MLP([obs_dim] + list(hidden_sizes) + [1], activation)
        self.v_net.apply(critic_init)

    def forward(self, obs):
        """
        Return the value estimate for a given observation.

        Parameters
        ----------
            obs: torch.Tensor of shape (n, obs_dim)
                State observation.

        Returns
        -------
            v: torch.Tensor of shape (n, ), i.e., where n is the
            number of observations. Value estimate for obs.
        """
        return th.squeeze(self.v_net(obs), -1)


class Operator:
    def __init__(self, env, obs_dim, act_dim,
                 actor_architecture=[32, 32],
                 critic_architecture=[128, 128],
                 actor_activation=nn.Tanh,
                 critic_activation=nn.Tanh):
        self.env = env
        self.actor = Actor(obs_dim, act_dim, actor_architecture, actor_activation)
        self.critic = Critic(obs_dim, critic_architecture, critic_activation)

    def step(self, state):
        """
        Take an state and return action, value function, and log-likelihood
        of chosen action.

        Parameters
        ----------
        state: torch.Tensor of shape (obs_dim, )

        Returns
        -------
        act: np.ndarray of (act_dim, )
            An action sampled from the policy given a state (0, 1, 2 or 3).
        v: np.ndarray of (1, )
            The value function at the given state.
        log_p: np.ndarray of (1, )
            The log-probability of the action under the policy distribution.
        """

        with th.no_grad():
            act = self.get_action(state)
            v = self.critic.forward(state)
            _, log_p = self.actor.forward(state, act)
        return act, v, log_p

    def get_action(self, obs):
        """
        Sample an action from your policy/actor.

        Parameters
        ----------
        obs: np.ndarray of shape (obs_dim, )
            State observation.

        Returns
        -------
        act: np.ndarray of shape (act_dim, )
            Action to apply.
        """

        pi = self.actor.forward(obs)[0]
        act = pi.sample()
        return act
