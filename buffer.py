__author__ = "Georgi Tancev, PhD"
__copyright__ = "© Georgi Tancev"

import numpy as np
import scipy.signal
import torch as th
import scipy.signal


def discount_cumsum(x, discount):
    """
    Compute cumulative sums of vectors.

    Input: [x0, x1, ..., xn]
    Output: [x0 + discount * x1 + discount^2 * x2 ... disounct^n * xn,
             x1 + discount * x2 ... discount^(n-1) * xn,
             ...,
             xn]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]


def combined_shape(length, shape=None):
    """
    Helper function that combines two array shapes.
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class Buffer:
    """
    Buffer to store trajectories.
    """
    def __init__(self, obs_dim, act_dim, size, gamma, lamda):
        # state space
        self.obs_buf = np.zeros(combined_shape(size, obs_dim),
                                dtype=np.float32)
        # action space
        self.act_buf = np.zeros(combined_shape(size, act_dim),
                                dtype=np.float32)
        # calculated TD residuals
        self.tdres_buf = np.zeros(size, dtype=np.float32)
        # rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # trajectory's remaining return
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # values predicted
        self.val_buf = np.zeros(size, dtype=np.float32)
        # log probabilities of chosen actions under behavior policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # hyperparameters for GAE
        self.gamma = gamma
        self.lamda = lamda
        # pointer to the latest data point in the buffer
        self.ptr = 0
        # pointer to the start of the trajectory
        self.path_start_idx = 0
        # maximum size of the buffer
        self.max_size = size

    def store(self, obs, act, rew, val, logp):
        """
        Append a single timestep to the buffer. This is called at
        each environment update to store the observed outcome in
            self.obs_buf,
            self.act_buf,
            self.rew_buf,
            self.val_buff,
            self.logp_buff.

        Parameters
        ----------
        obs: torch.Tensor of shape (obs_dim, )
            State observation.

        act: torch.Tensor of shape (act_dim, )
            Applied action.

        rew: torch.Tensor of shape (1, )
            Observed rewards.

        val: torch.Tensor of shape (1, )
            Predicted values.

        logp: torch.Tensor of shape (1, )
            log probability of act under behavior policy
        """

        # buffer has to have room so you can store
        assert self.ptr < self.max_size

        self.obs_buf[self.ptr, :] = obs
        self.act_buf[self.ptr, :] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp

        # Update pointer after data is stored.
        self.ptr += 1

    def end_traj(self, last_val=0):
        """
        Calculate for a trajectory
            1) discounted rewards-to-go, and
            2) TD residuals.
        Store these into self.ret_buf, and self.tdres_buf respectively.

        The function is called after a trajectory ends.

        Parameters
        ----------
        last_val: np.float32
            Last value is value (state) if the rollout is cut-off at a
            certain state, or 0 if trajectory ended uninterrupted.
        """

        # Get the indexes where TD residuals and discounted
        # rewards-to-go are stored.
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        delta = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.tdres_buf[path_slice] = discount_cumsum(delta,
                                                     self.lamda * self.gamma)

        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        # Update the path_start_idx
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call after an epoch ends.
        Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        tdres_mean = np.mean(self.tdres_buf)
        tdres_std = np.std(self.tdres_buf)
        self.tdres_buf = (self.tdres_buf - tdres_mean) / (tdres_std)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    val=self.val_buf, tdres=self.tdres_buf, logp=self.logp_buf)
        return {k: th.as_tensor(v, dtype=th.float32)
                for k, v in data.items()}
