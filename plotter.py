__author__ = "Georgi Tancev, PhD"
__copyright__ = "© Georgi Tancev"

import os
import numpy as np
import matplotlib.pyplot as plt
from environment import BatchCoolingCrystallization
plt.rcParams["lines.linewidth"] = 1.0
plt.rcParams["font.size"] = 6.0
plt.rcParams["axes.titlesize"] = 6.0


class Plotter:
    """
    Plotter class.
    """
    def __init__(self,
                 obs_dim, act_dim,
                 episode_length, n_eval=50,
                 path="current_run"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.episode_length = episode_length
        self.n_eval = n_eval
        self.path = path+"/snapshots"

        # Check if path exists, and if not, create it.
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    @staticmethod
    def moving_average(s, a=0.10):
        """
        Exponential moving average.

        a: smoothing factor in [0, 1].
        """
        T = len(s)
        r = np.zeros(T)
        r[0] = s[0]
        for k in range(1, T):
            r[k] = (1 - a) * r[k-1] + a * (s[k])
        return r

    def evaluate(self, env, operator, normalizer):
        """
        Function to evaluate the current policy.
        """

        obs_dim = self.obs_dim
        act_dim = self.act_dim
        n_eval = self.n_eval
        episode_length = self.episode_length

        trajectories = np.zeros((n_eval, episode_length - 1, obs_dim + act_dim))
        for i in range(n_eval):
            state = env.reset()
            for t in range(episode_length - 1):
                action = operator.get_action(normalizer.transform(state))
                trajectories[i, t, :] = np.concatenate((state,
                                                        action.numpy().ravel()))
                state, _, terminal = env.transition(state, action.numpy().ravel())
                if terminal:
                    break

        return trajectories

    def take_snapshot(self, env, operator, normalizer, epoch, save_trajectory=True):
        """
        Function to draw figures.
        """

        path = self.path
        trajectories = self.evaluate(env, operator, normalizer)
        if save_trajectory:
            np.save(path+"/trajectory_epoch_"+str(epoch), np.mean(trajectories, axis=0))
        _, T, _ = trajectories.shape
        env_vars = vars(env) # get environment variables
        time_delta = env_vars["time_delta"] # min

        if isinstance(env, BatchCoolingCrystallization):

            # Draw states.
            L = env_vars["L"] # length, μm
            k_V = env_vars["k_V"] # shape factor, -
            rho = env_vars["rho"] # density of crystals, kg / m ** 3
            t = time_delta * np.arange(0, T) / 60.0
            t_max = np.round(self.episode_length * time_delta / 60.0, 0) # h
            n = trajectories[:, :, :-3]
            
            fig, axes = plt.subplots(2, 3, sharex=True,
                                    figsize=(2.4*2.90, 0.6*2.90))
            fig.tight_layout(w_pad=2.0)
            
            ax0 = axes.ravel()[0]
            m_s = np.trapz(np.mean(n, axis=0) * (L * 1e-6) ** 3, L) * k_V * rho
            y = self.moving_average(m_s, a=0.5)
            ax0.plot(t, y, color=plt.cm.viridis(0.0))
            ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax0.set_ylim(0, 0.2)
            ax0.set_yticks(np.linspace(0, 0.2, 3))
            ax0.set_xlim(0, t_max)
            ax0.set_xticks(np.linspace(0, t_max, 4))
            ax0.set_ylabel(r'$m$ / $\frac{kg}{kg}$', labelpad=5)
            ax0.minorticks_on()
            ax0.grid(which="both", alpha=0.1)
            # ax0.set_xlabel(r"$t$ / $h$")

            ax1 = axes.ravel()[3]
            L_V = (np.sum(n * (L) ** 4, axis=-1)) / (np.sum(n * (L) ** 3, axis=-1) + 1.0e0)
            y = self.moving_average(np.mean(L_V, axis=0), a=0.5)
            ax1.plot(t, y, label=r"$\bar{d}_p$", color=plt.cm.viridis(0.15))
            ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax1.set_ylim(0, 800)
            ax1.set_yticks(np.linspace(0, 800, 3))
            ax1.set_xlim(0, t_max)
            ax1.set_xticks(np.linspace(0, t_max, 4))
            ax1.set_ylabel(r"$\bar{d}$ / $\mu m$", labelpad=7)
            ax1.set_xlabel(r"$t$ / $h$")
            ax1.minorticks_on()
            ax1.grid(which="both", alpha=0.1)

            ax4 = axes.ravel()[1]
            c0 = env_vars["c_0"]
            c = c0 - m_s
            y = self.moving_average(c, a=0.5)
            ax4.plot(t, y, color=plt.cm.viridis(0.30))
            ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax4.set_ylim(0.1, 0.3)
            ax4.set_yticks(np.linspace(0.1, 0.3, 3))
            ax4.set_ylabel(r'$c$ / $\frac{kg}{kg}$', labelpad=7)
            ax4.minorticks_on()
            ax4.grid(which="both", alpha=0.1)

            ax5 = axes.ravel()[4]
            T = np.mean(trajectories[:, :, -3], axis=0)
            c_s = env.c_s(T)
            y = self.moving_average(c / c_s, a=0.5)
            ax5.plot(t, y, color=plt.cm.viridis(0.45))
            ax5.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax5.set_ylim(0, 2)
            ax5.set_yticks(np.linspace(0, 2, 3))
            ax5.set_ylabel(r'$S$ / $-$', labelpad=9)
            ax5.set_xlabel(r"$t$ / $h$")
            ax5.minorticks_on()
            ax5.grid(which="both", alpha=0.1)

            ax2 = axes.ravel()[2]
            y = self.moving_average(T - 273.15 , a=0.5)
            ax2.plot(t, y, label="$T$", color=plt.cm.viridis(0.60))
            ax2.set_ylim(0, 80)
            # ax2.set_yscale("log")
            ax2.set_yticks(np.linspace(0, 80, 3))
            ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax2.set_ylabel(r"$T$ / $°C$", labelpad=10)
            ax2.minorticks_on()
            ax2.grid(which="both", alpha=0.1)

            # Draw actions.
            Q_max = env_vars["Q_max"]
            A_max = np.array([(2.0e0 * Q_max)])
            ax3 = axes.ravel()[5]
            y_raw = np.mean(trajectories[:, :, -1], axis=0) * A_max[0] - Q_max
            y = self.moving_average(y_raw)
            ax3.plot(t, y, label="$H$", color=plt.cm.viridis(0.75))
            ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax3.set_xlabel(r"$t$ / $h$")
            ax3.set_ylabel(r"$H$ / $\frac{°C}{min}$")
            ax3.set_xlim(0, t_max)
            ax3.set_xticks(np.linspace(0, t_max, 4))
            ax3.set_ylim(-1.0, 1.0)
            ax3.set_yticks(np.linspace(-1.0, 1.0, 3))
            ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax3.minorticks_on()
            ax3.grid(which="both", alpha=0.1)
        
        plt.savefig(path+"/epoch_"+str(epoch)+".png",
                    dpi=1200, transparent=False, orientation="landscape",
                    bbox_inches="tight")
        plt.close(fig)

        if isinstance(env, BatchCoolingCrystallization):

            c_f = 1e18 # conversion factor, μm3 / m3
            k_V = env_vars["k_V"] # shape factor, -
            rho = env_vars["rho"] # density of crystals, kg / m ** 3
            L = env_vars["L"]
            p_V = np.mean(trajectories[:, -1, :-3], axis=0) * (L * 1e-6) ** 3 * c_f * k_V * rho

            fig, ax = plt.subplots(1, 1, sharex=True, figsize=(1.2*2.95, 0.4*2.95))
            ax.plot(L, p_V, color=plt.cm.viridis(0.25))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(10, 10000)
            ax.set_ylim(1e8, 1e16)
            # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.set_xlabel(r'$d$ / $\mu m$')
            ax.set_ylabel(r'$\rho_v$ / $\frac{\mu m^3}{\mu m \cdot kg}$')
            ax.minorticks_on()
            ax.grid(which="both", alpha=0.1)
            plt.savefig(path+"/psd_epoch_"+str(epoch)+".png", 
                        dpi=1200, transparent=False, 
                        orientation='landscape', bbox_inches="tight")
            plt.close(fig)

