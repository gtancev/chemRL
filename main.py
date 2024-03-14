__author__ = "Georgi Tancev, PhD"
__copyright__ = "© Georgi Tancev"

import os
import random
from datetime import datetime
import pickle
import numpy as np
import torch as th
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from environment import BatchCoolingCrystallization
from buffer import Buffer
from agent import Operator
from plotter import Plotter
from helper import OnlineStandardizer


def train(n_epochs=5000,
          steps_per_epoch=5000,
          max_ep_len=1440,
          actor_lr=3.0e-4,
          actor_lr_decay=9.995e-1,
          actor_updates=10,
          actor_max_grad_norm=5.0e-1,
          critic_lr=3.0e-4,
          critic_lr_decay=9.995e-1,
          critic_updates=10,
          critic_max_grad_norm=5.0e-1,
          optimizer_betas=(9.0e-1, 9.0e-1),
          discount_factor=9.9e-1,
          trace_decay_factor=9.0e-1,
          critic_clipping_parameter=2.0e1,
          actor_clipping_parameter=2.0e-1,
          entropy_coefficient=1.0e-5,
          KL_threshold=3.0e-3,
          obs_normalization=False,
          seed=3,
          snapshot=100,
          path="runs"):
    """
    Main training loop.

    n_epochs: number of epochs to train for
    steps_per_epoch: number of training steps per epoch
    max_ep_len: the longest an episode can go on before cutting it off
    lr: learning rate
    lr_decay: decay rate of learning rate
    n_updates: number of gradient steps
    max_grad_norm: maximum gradient norm
    optimizer_betas: parameters (as tuple) for Adam
    discount_factor: gamma
    trace_decay_factor: lambda
    critic_clipping_parameter: epsilon
    actor_clipping_parameter: epsilon
    entropy_coefficient: alpha
    KL_threshold: at what KL divergence to stop updates
    obs_normalization: whether to normalize state space in an online manner
    seed: seed
    snapshot: after how many epochs to take a snapshot
    path: name of folder for experimental results
    """

    # Set seeds.
    th.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Catch input arguments.
    input_args = locals()

    # Initialize environment.
    env = BatchCoolingCrystallization(n_transitions=max_ep_len)
    env_vars = vars(env)
    obs_dim, act_dim = env_vars["obs_dim"], env_vars["act_dim"]

    # Initialize normalizer.
    normalizer = OnlineStandardizer(obs_dim,
                                    use_normalization=obs_normalization)

    # Initialize operator.
    agent = Operator(env, obs_dim, act_dim)

    # Set up buffer.
    buf = Buffer([obs_dim], [act_dim], steps_per_epoch, discount_factor, trace_decay_factor)

    # Set up folders/arrays and store (meta)data.
    summary = {"returns": np.zeros((n_epochs)),
               "loss_actor": np.zeros((n_epochs)),
               "KL divergence": np.zeros((n_epochs)),
               "loss_critic": np.zeros((n_epochs))}
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path += "/"+time
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"/exp_settings.pkl", "wb") as df:
        pickle.dump(input_args, df)
    with open(path+"/env_settings.pkl", "wb") as df:
        pickle.dump(env_vars, df)
    
    # Set up storage and plotter.
    if snapshot is not None:
        plotter = Plotter(obs_dim, act_dim, max_ep_len, 
                          path=path)

    # Initialize the optimizer using the parameters
    # of the actor and then critic networks.
    actor_optimizer = Adam(agent.actor.parameters(),
                           lr=actor_lr,
                           betas=optimizer_betas)
    critic_optimizer = Adam(agent.critic.parameters(),
                            lr=critic_lr,
                            betas=optimizer_betas)
    if actor_lr_decay is not None:
        actor_scheduler = ExponentialLR(actor_optimizer,
                                        actor_lr_decay)
    if critic_lr_decay is not None:
        critic_scheduler = ExponentialLR(critic_optimizer,
                                         critic_lr_decay)

    # Initialize the environment.
    state, ep_ret, ep_len = agent.env.reset(), 0, 0

    # Main training loop: collect experience in env and update/log each epoch.
    for epoch in range(n_epochs):

        # Reset returns.
        ep_returns = []

        for t in range(steps_per_epoch):

            # Perform step.
            a, v, log_p = agent.step(th.as_tensor(normalizer.transform(state),
                                                  dtype=th.float32))
            next_state, r, terminal = agent.env.transition(state,
                                                           a.numpy().ravel())
            ep_ret += r
            ep_len += 1

            # Log transition.
            buf.store(state, a, r, v, log_p)

            # Update state.
            state = next_state

            timeout = (ep_len == max_ep_len)
            epoch_ended = (t == steps_per_epoch - 1)

            if terminal or timeout or epoch_ended:
                # If trajectory didn't reach terminal state,
                # bootstrap value target.
                if epoch_ended:
                    _, v, _ = agent.step(th.as_tensor(normalizer.transform(state),
                                                      dtype=th.float32))
                else:
                    v = 0
                # Only store return if episode ended.
                if timeout or terminal:
                    ep_returns.append(ep_ret)
                buf.end_traj(v)
                state, ep_ret, ep_len = agent.env.reset(), 0, 0

        # Calculate average reward.
        mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan

        # Get data from buffer.
        data = buf.get()
        obs = data["obs"]
        act = data["act"]
        target = data["ret"]
        values_old = data["val"]
        tdres = data["tdres"]
        log_p_old = data["logp"]

        # Update normalizer and normalize observations.
        if obs_normalization:
            normalizer.fit(obs.numpy())
            obs = th.as_tensor(normalizer.transform(obs), dtype=th.float32)

        # Perform policy gradient updates.
        for _ in range(actor_updates):
            actor_optimizer.zero_grad()
            pi, log_p = agent.actor.forward(obs, act)
            ratio = th.exp(log_p - log_p_old)
            gain = tdres * ratio
            bound = tdres * th.clip(ratio,
                                    1 - actor_clipping_parameter,
                                    1 + actor_clipping_parameter)
            importance = th.min(gain, bound).mean()
            entropy = pi.entropy().mean()
            KL_divergence = ((ratio - 1) - th.log(ratio)).mean()
            loss_actor = - importance - entropy_coefficient * entropy
            loss_actor.backward()

            if actor_max_grad_norm is not None:
                clip_grad_norm_(agent.actor.parameters(),
                                actor_max_grad_norm)

            actor_optimizer.step()

            if (KL_divergence >= KL_threshold):
                break

        # Perform value function updates.
        for _ in range(critic_updates):
            critic_optimizer.zero_grad()
            values = agent.critic.forward(obs)
            squared_residuals = (values_old + th.clip(values - values_old,
                                                      - critic_clipping_parameter,
                                                      + critic_clipping_parameter) - target).square().mean()
            loss_critic = squared_residuals
            loss_critic.backward()
            if critic_max_grad_norm is not None:
                clip_grad_norm_(agent.critic.parameters(),
                                critic_max_grad_norm)
            critic_optimizer.step()

        # Perform scheduler steps.
        if actor_lr_decay is not None:
            actor_scheduler.step()
        if critic_lr_decay is not None:
            critic_scheduler.step()

        # Store losses.
        summary["returns"][epoch] = mean_return
        summary["loss_actor"][epoch] = loss_actor.detach().numpy()
        summary["KL divergence"][epoch] = KL_divergence.detach().numpy()
        summary["loss_critic"][epoch] = loss_critic.detach().numpy()

        # Print progress.
        print(f"Epoch: {epoch+1}/{n_epochs}, "
              f"average return: {np.round(mean_return, 2)}, "
              f"actor loss: {th.round(-loss_actor, decimals=4).detach().numpy()}, "
              f"Kullback-Leiber divergence: {th.round(KL_divergence, decimals=4).detach().numpy()}, "
              f"critic loss: {th.round(loss_critic, decimals=2).detach().numpy()}.")
        
        # Draw a plot.
        if (snapshot is not None) and ((epoch + 1) % snapshot == 0):
            plotter.take_snapshot(env, agent, normalizer, epoch + 1)

    # Save learning curve.
    with open(path+"/summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    return agent


if __name__ == "__main__":
    operator = train()
