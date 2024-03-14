__author__ = "Georgi Tancev, PhD"
__copyright__ = "© Georgi Tancev"

import torch.nn as nn
import numpy as np


def actor_init(w, gain=1.0e0):
    """
    Function to initialize weights of actor.
    """
    if type(w) == nn.Linear:
        nn.init.xavier_uniform_(w.weight, gain)


def critic_init(w, gain=1.0e0):
    """
    Function to initialize weights of critic.
    """
    if type(w) == nn.Linear:
        nn.init.xavier_uniform_(w.weight, gain)


class OnlineStandardizer(object):
    """
    Class that keeps track of mean and variance across the dimensions
    during learning.
    """
    def __init__(self, obs_dim, epsilon=1.0e-4, use_normalization=False):
        self.mean = np.zeros((1, obs_dim), "float64")
        self.var = np.ones((1, obs_dim), "float64")
        self.count = epsilon
        self.use_normalization = use_normalization

    @staticmethod
    def update(mean, var, count, batch_mean, batch_var, batch_count):
        """
        Method that recomputes mean and variance.
        """
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * (batch_count / tot_count)
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * (batch_count / tot_count)
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    def fit(self, obs):
        """
        Method that updates mean and variance estimates.
        """
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        batch_count = obs.shape[0]
        self.mean, self.var, self.count = self.update(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def transform(self, obs):
        """
        Method that scales observations (zero mean, unit variance).
        """
        if self.use_normalization:
            return np.clip((obs - self.mean) / np.clip(np.sqrt(self.var), 1.0e-2, np.inf), -1.0e1, 1.0e1)
        else:
            return obs

