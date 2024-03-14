__author__ = "Georgi Tancev, PhD"
__copyright__ = "© Georgi Tancev"

import numpy as np
from scipy.integrate import odeint


class BatchCoolingCrystallization:
    """
    Environment class for batch cooling crystallization 
    that describes the dynamics of the system
    and contains a method to perform transitions
    and determine rewards.
    """
    def __init__(self,
                 n_transitions=1440, time_delta=5.0e-1,
                 T_min=1.0e1, T_max=7.0e1,
                 T_crit=1.0e1, Q_max=1.0e0,
                 T_0=6.0e1, c_0=2.5e-1,
                 dL=5.0e0, L_max=2.0e3,
                 M=1.512e-1, rho=1.3e3, k_V=7.97e-1,
                 reward_scaling=1.0e-2):
        self.t_max = n_transitions * time_delta # total operating time in min
        self.time_delta = time_delta # time passed per transition in min
        self.T_min = T_min + 273.15 # minimum reactor temperature in °C
        self.T_max = T_max + 273.15 # maximum reactor temperature in °C
        self.T_crit = T_crit # margin for termperature in K
        self.Q_max = Q_max # heating/cooling power in °C/min
        self.T_0 = T_0 + 273.15 # initial Temperature in K
        self.c_0 = c_0 # initial state (concentration) in kg-solute/kg-solvent
        self.L_max = L_max # maximum length in μm
        self.dL = dL # bin size in μm
        self.L = np.arange(dL, L_max + dL, dL) # length in μm
        self.M = M # molar mass of solute, kg / mol
        self.rho = rho # density of crystals, kg / m ** 3
        self.k_V = k_V # shape factor, -
        self.reward_scaling = reward_scaling # reward scaling
        self.obs_dim = self.L.shape[0] + 2 # dimensionality of state space vector
        self.act_dim = 1 # dimensionality of action space vector

    @staticmethod
    def c_s(T):
        """
        Solubility curve.

        Input
        -----
        T: temperature in K

        Output
        ------
        c_s: solubility, kg-solute/kg-solvent
        """
        return -8.707 + 9.669e-2 * T - 3.610e-4 * T ** 2 + 4.590e-7 * T ** 3

    @staticmethod
    def dynamics(y, t, action, c_0, L, dL, Q_max, M, rho, k_V, c_s, time_delta):
        """
        System of differential equations for the dynamics model.

        Input
        -----
        y: state space, a.u.
        t: time, min
        action: set of actions
        c_0: initial concentration of solute, kg-solute/kg-solvent
        L: bin vector, μm
        dL: bin size, μm
        Q_max: maximum heating/cooling power, °C/min
        M: molar mass of solute, kg / mol
        rho: density of crystals, kg / m ** 3
        k_V: shape factor, -
        c_s: solubility function, kg-solute/kg-solvent
        time_delta: time passed per iteration

        Output
        ------
        dydt: set of differential equations in a list
        """

        # Define constants.
        k_d = -4.08e4 # pre-exponential rate constant for dissolution, (μm/min)(kg-solute/kg-solvent) ** −gamma_d
        E_ad = 9.8e3 # activation energy for dissolution, J / mol
        gamma_d = 9.3e-1 # exponential parameters on supersaturation for dissolution, -

        k_g = 2.26e8 # pre-exponential rate constant for growth, (μm/min)(kg-solute/kg-solvent) ** −gamma_d
        E_ag = 3.62e4 # activation energy for growth, J / mol
        gamma_g = 1.14e0 # exponential parameters on supersaturation for growth, -

        k_b1 = 1.06e1 # rate constant for primary nucleation, #/(min·kg solvent)
        k_b2 = 1.39e6 # rate constant for secondary nucleation, #/(min·kg solvent)

        alpha = 2.39e0 # exponential parameter for the model, -
        beta = 4.1e-1 # exponential parameter for the model, -
        sigma = 3.83e-3 # interfacial energy between crystal and solution, J / m ** 2
        R = 8.3144e0 # gas constant, J / (K⋅mol)
        k_B = 1.380649e-23 # Boltzmann constant, J / K

        N_A = 6.02214076e23 # Avogradro constant, 1 / mol
        nu = (M / N_A) / rho # volume of one solute molecule, m ** 3

        # Get states.
        n = np.clip(y[:-2], 0.0e0, 1.0e6) # number concentration of crystals, # / (μm·kg of solvent)
        T = y[-2] # temperature, K

        # Get actions.
        Q = (2.0e0 * Q_max) * action[0] - Q_max # heating or cooling, K / min
        # Q = action[0]

        # Compute total mass of crystals.
        m_s = np.trapz(n * (L * 1e-6) ** 3, L) * k_V * rho # mass concentration of crystals, kg-crystals/kg-solvent

        # Get supersaturation.
        c_s = c_s(T) # solubility, kg-solute/kg-solvent
        c = c_0 - m_s # mass concentration in solution, kg-solute/kg-solvent
        S = c / c_s # supersaturation, -

        # Set growth and dissolution rates.
        if S < 1.0e0: # undersaturation
            G = k_d * np.exp(- E_ad / (R * T)) * (c_s - c) ** gamma_d # dissolution rate of crystals, μm/min
        else: # supersaturation
            G = k_g * np.exp(- E_ag / (R * T)) * (c - c_s) ** gamma_g # growth rate of crystals, μm/min

        # Set nucleation rates.
        if S < 1.0e0: # undersaturation
            B_1, B_2 = 0.0e0, 0.0e0 # nucleation rates, # / (min·kg of solvent)
        else: # supersaturation
            B_1 = k_b1 * np.exp(-(16 * np.pi * nu ** 2 * sigma ** 3) / (3 * k_B ** 3 * T ** 3 * np.log(S) ** 2)) # primary nucleation rate, # / (min·kg of solvent)
            B_2 = k_b2 * (S - 1) ** alpha * m_s ** beta # secondary nucleation rate, # / (min·kg of solvent)
        B = B_1 + B_2

        # Compute derivatives for number concentrations.
        if S < 1.0e0: # undersaturation
            n0 = 0.0e0 # boundary condition (at L = 0)
            dydt = [] # initialize list
            for i in range(0, len(n) - 1):
                dydt.append((G / dL) * (n[i] - n[i + 1])) # flow from above
            dydt.append((G / dL) * n[-1]) # derivative for largest size

        else: # supersaturation
            n0 = B / G # # boundary condition (at L = 0)
            dydt = [(G / dL) * (n0 - n[0])] # derivative for smallest size
            for i in range(1, len(n)):
                dydt.append((G / dL) * (n[i - 1] - n[i])) # flow from below
        
        # Describe temperature change.
        dydt.append(Q)

        # Describe operating time index change.
        dydt.append(1.0e0 / time_delta)

        return dydt


    def get_reward(self, last_state, action, next_state):
        """
        Method that calculates the reward (i.e., reward function).

        Input
        -----
        last_state: state before transition
        action: set of actions
        next_state: state after transition

        Output
        ------
        r: reward for the transition
        terminal: if a terminal state has been reached
        """

        # Initialize reward and terminal flag.
        r = 1.0e0
        terminal = False

        # Get states.
        n, n_l = next_state[:-2], last_state[:-2] # number concentration, # / (μm·kg of solvent)
        T = next_state[-2] # temperature, K
        t = next_state[-1] # operating time, min

        # Get actions.
        Q_max = self.Q_max # maximum heat increase, K / min
        Q = (2.0e0 * Q_max) * action[0] - Q_max # heating/cooling, K / min
        # Q = action[0]

        # Shape rewards.
        L, L_max, T_min, T_max, T_crit = self.L, self.L_max, self.T_min, self.T_max, self.T_crit
        k_V, rho, c_0, c_s_min = self.k_V, self.rho, self.c_0, self.c_s(T_min)
        t_max = self.t_max

        L_v_l = (np.sum(n_l * (L) ** 4)) / (np.sum(n_l * (L) ** 3) + 1.0e8) # mean diameter, μm
        L_v = (np.sum(n * (L) ** 4)) / (np.sum(n * (L) ** 3) + 1.0e8) # mean diameter, μm
        m_s_l = (np.trapz(n_l * (L * 1e-6) ** 3, L) * k_V * rho) # mass concentration of crystals, kg-crystals/kg-solvent
        m_s = (np.trapz(n * (L * 1e-6) ** 3, L) * k_V * rho) # mass concentration of crystals, kg-crystals/kg-solvent

        if (L_v <= 0.0e0): # if no crystals present in next state
            r -= 1.0e0 * (Q / Q_max) # reward cooling but penalize heating
        if (L_v > 0.0e0): # if crystals present in next state
            r += 1.0e0 # base reward for producing and mantaining particles
            r += 2.0e1 * (L_v / L_max) # particle size
            r += 2.0e3 * ((L_v - L_v_l) / L_max) # particle size
            # r += 1.0e3 * ((m_s - m_s_l) / (c_0 - c_s_min)) # yield
            r -= 1.0e-1 * (Q / Q_max) * (L_v / L_max) # promote cooling (but much less)

        # Find terminal states (and penalize).
        if (T <= T_min):
            # r -= 1.0e0
            r -= 1.0e1 * np.abs((T - T_min) / T_crit)
            # r += 1.0e0 * (Q / Q_max) # reward heating but penalize cooling
        if (T >= T_max):
            # r -= 1.0e0
            r -= 1.0e1 * np.abs((T - T_max) / T_crit)
            # r -= 1.0e0 * (Q / Q_max) # reward cooling but penalize heating
        if (T <= T_min - T_crit) or (T >= T_max + T_crit):
            terminal = True

        # Rescale reward.
        r *= self.reward_scaling

        return r, terminal

    def reset(self):
        """
        Method that defines the initial state of the system.

        Output
        ------
        initial_state: state at t = 0
        """
        initial_state = np.zeros((self.obs_dim)) # initial number concentrations
        initial_state[-2] = self.T_0 # initial temperature, K

        return initial_state

    def transition(self, current_state, action):
        """
        Perform one transition of predefined duration.

        Input
        -----
        current_state: state before transition
        action: actions in that state

        Output
        ------
        next_state: state after transition
        reward: reward for transition
        terminal: if a terminal state has been reached
        """

        # Integrate differential equations (and retain last values).
        c_0, L, dL, Q_max = self.c_0, self.L, self.dL, self.Q_max
        M, rho, k_V, c_s = self.M, self.rho, self.k_V, self.c_s
        time_delta = self.time_delta
        next_state = odeint(self.dynamics,
                            current_state,
                            np.linspace(0.0e0, self.time_delta),
                            args=(action, c_0, L, dL, Q_max,
                                  M, rho, k_V, c_s,
                                  time_delta))[-1, :]
        next_state = np.clip(next_state, 0.0e0, 1.0e6)

        # Get reward.
        r, terminal = self.get_reward(current_state,
                                      action,
                                      next_state)

        return next_state, r, terminal
