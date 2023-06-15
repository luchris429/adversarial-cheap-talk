import jax
import jax.numpy as jnp
import optax

import flax
from flax import linen as nn
import jax.numpy as jnp
from typing import NamedTuple, Callable, Any
from flax.training.train_state import TrainState

from tensorflow_probability.substrates import jax as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors


class PPO_Network(nn.Module):

    num_outputs: int
    hsize: int
    activation_fn: Callable[..., Any]
    discrete: bool = True
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hsize, name="hidden")(x)
        x = self.activation_fn(x)
        x = nn.Dense(features=self.hsize, name="hidden_2")(x)
        x = self.activation_fn(x)
        v = nn.Dense(1, name="vals")(x)
        out1 = nn.Dense(features=self.num_outputs, name="logits")(x)
        if self.discrete:
            pi = tfd.Categorical(logits=out1)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.num_outputs,))  # TODO: MAYBE JUST ONE SCALAR?
            log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
            pi = tfd.MultivariateNormalDiag(loc=out1, scale_diag=jnp.exp(log_stds))

        return v, pi


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def loss_actor_and_critic(
    params_model: flax.core.frozen_dict.FrozenDict,
    apply_fn: Callable[..., Any],
    state: jnp.ndarray,
    target: jnp.ndarray,
    value_old: jnp.ndarray,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    action: jnp.ndarray,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> jnp.ndarray:
    value_pred, pi = apply_fn(params_model, state)
    value_pred = value_pred[:, 0]

    log_prob = pi.log_prob(action)

    value_pred_clipped = value_old + (value_pred - value_old).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    total_loss = loss_actor + critic_coeff * value_loss - entropy_coeff * entropy

    return total_loss, (value_loss, loss_actor, entropy, value_pred.mean(), target.mean(), gae.mean())


class PPO:
    def __init__(self, network, env_reset, env_step, config):
        self.network = network
        # self.env_reset = jax.vmap(env.reset)
        # self.env_step = jax.vmap(env.step)
        self.env_reset = env_reset
        self.env_step = env_step
        self.config = config

    def _init_state(self, rng):
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(self.config["OBS_SIZE"])
        network_params = self.network.init(_rng, x=init_x)
        tx = optax.chain(optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]), optax.adam(self.config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=network_params,
            tx=tx,
        )
        return (train_state, rng)

    def _eval_step(self, val):
        network_params, env_state, last_obs, cumr, rng, ever_done = val
        action = self._select_action_deterministic(last_obs, network_params)
        env_state, obsv, reward, done, info = self.env_step(env_state, action)
        cumr = cumr + jnp.logical_not(ever_done) * reward
        ever_done = jnp.logical_or(done, ever_done)
        return (network_params, env_state, obsv, cumr, rng, ever_done)

    def eval_network(self, network_params, rng):
        rng, _rng = jax.random.split(rng)
        _env_rng = jax.random.split(_rng, self.config["NUM_EVAL_ENVS"])
        env_state, obsv = self.env_reset(_env_rng)
        val = (network_params, env_state, obsv, jnp.zeros(self.config["NUM_EVAL_ENVS"]), rng, jnp.zeros(self.config["NUM_EVAL_ENVS"], dtype=bool))
        val = jax.lax.while_loop(lambda v: jnp.any(jnp.logical_not(v[-1])), self._eval_step, val)
        return val[3]

    # @partial(jit, static_argnums=(0,))
    def train(self, rng, train_state=None):
        # INIT NETWORKS
        if train_state is None:
            train_state, rng = self._init_state(rng)

        # INIT ENV
        all_rng = jax.random.split(rng, self.config["NUM_ENVS"] + 1)
        rng, _rng = all_rng[0], all_rng[1:]
        env_state, obsv = self.env_reset(_rng)

        # TRAIN LOOP
        #         for _ in range(self.config["NUM_UPDATES"]):
        #             val, _ = self._update_step(val, None)
        # all_rng = jax.random.split(rng, self.config["NUM_ENVS"])
        # rng, _rng = all_rng[0], all_rng[1:]
        rng, _rng = jax.random.split(rng)
        val = (train_state, env_state, obsv, _rng)
        val, metric = jax.lax.scan(self._update_step, val, None, self.config["NUM_UPDATES"])

        return val, metric

    def _get_traj_batch(self, val):
        val, traj_batch = jax.lax.scan(self._env_step, val, None, self.config["UPDATE_PERIOD"])
        train_state = val[0]
        advantages, targets = self._calculate_gae(traj_batch)
        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        if self.config["DISCRETE"]:
            action = traj_batch.action[:-1].reshape((-1,))
        else:
            action = traj_batch.action[:-1].reshape((-1, self.config["NUM_ACTIONS"]))

        total_loss, grads = grad_fn(
            train_state.params,
            self.network.apply,
            state=traj_batch.obs[:-1].reshape((-1, self.config["OBS_SIZE"])),
            target=targets.reshape((-1,)),
            value_old=traj_batch.value[:-1].reshape((-1,)),
            log_pi_old=traj_batch.log_prob[:-1].reshape((-1,)),
            gae=advantages.reshape((-1,)),
            action=action,
            clip_eps=self.config["CLIP_EPS"],
            critic_coeff=self.config["CRITIC_COEFF"],
            entropy_coeff=self.config["ENTROPY_COEFF"],
        )

        train_state = train_state.apply_gradients(grads=grads)
        val = (train_state, *val[1:])
        return val, (total_loss, grads)

    def _update_step(self, val, unused):
        val, traj_batch = jax.lax.scan(self._env_step, val, None, self.config["UPDATE_PERIOD"])
        new_train_state, info = self._update(val[0], traj_batch)
        metric = traj_batch.reward.sum() / (traj_batch.done.sum())
        if self.config["GEN_EXPLOIT"]:
            outer_metric = traj_batch.info.sum() / (traj_batch.done.sum())
            metric = (metric, outer_metric)
        val = (new_train_state, *val[1:])
        if self.config["DEBUG"]:
            metric = (metric, info)
        else:
            return val, metric

    def _env_step(self, val, unused):
        train_state, env_state, last_obs, rng = val
        action, log_prob, value, rng = self._select_action(last_obs, train_state.params, rng)
        env_state, obsv, reward, done, info = self.env_step(env_state, action)
        transition = Transition(
            done=done,
            action=action,
            value=value,
            reward=reward,
            log_prob=log_prob,
            obs=last_obs,
            info=info,
        )
        val = (train_state, env_state, obsv, rng)
        return val, transition

    def _select_action_deterministic(self, obs, network_params):
        value, pi = self.network.apply(network_params, obs)
        action = pi.mode()
        return action

    def _select_action(self, obs, network_params, rng):
        value, pi = self.network.apply(network_params, obs)
        rng, key = jax.random.split(rng)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, log_prob, value[:, 0], rng

    def _update(self, train_state, traj_batch):
        advantages, targets = self._calculate_gae(traj_batch)
        # val = (traj_batch, advantages, targets, train_state)  # TODO: PARTIAL IN THE BATCH? DOES THIS MATTER? IDEK.
        # val, info = jax.lax.scan(self._update_epoch, val, None, self.config["NUM_EPOCHS_PER_UPDATE"])

        _update_epoch_batch = self._mk_update_epoch(traj_batch, advantages, targets)
        train_state, info = jax.lax.scan(_update_epoch_batch, train_state, None, self.config["NUM_EPOCHS_PER_UPDATE"])

        # return val[-1], info
        return train_state, info

    def _mk_update_epoch(self, traj_batch, advantages, targets):
        def _update_epoch_partial(train_state, unused):
            grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
            if self.config["DISCRETE"]:
                action = traj_batch.action[:-1].reshape((-1,))
            else:
                action = traj_batch.action[:-1].reshape((-1, self.config["NUM_ACTIONS"]))

            total_loss, grads = grad_fn(
                train_state.params,
                self.network.apply,
                state=traj_batch.obs[:-1].reshape((-1, self.config["OBS_SIZE"])),
                target=targets.reshape((-1,)),
                value_old=traj_batch.value[:-1].reshape((-1,)),
                log_pi_old=traj_batch.log_prob[:-1].reshape((-1,)),
                gae=advantages.reshape((-1,)),
                action=action,
                clip_eps=self.config["CLIP_EPS"],
                critic_coeff=self.config["CRITIC_COEFF"],
                entropy_coeff=self.config["ENTROPY_COEFF"],
            )

            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        return _update_epoch_partial

    def _update_epoch(self, val, unused):
        batch, gae, targets, train_state = val
        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        if self.config["DISCRETE"]:
            action = batch.action[:-1].reshape((-1,))
        else:
            action = batch.action[:-1].reshape((-1, self.config["NUM_ACTIONS"]))

        total_loss, grads = grad_fn(
            train_state.params,
            self.network.apply,
            state=batch.obs[:-1].reshape((-1, self.config["OBS_SIZE"])),
            target=targets.reshape((-1,)),
            value_old=batch.value[:-1].reshape((-1,)),
            log_pi_old=batch.log_prob[:-1].reshape((-1,)),
            gae=gae.reshape((-1,)),
            action=action,
            clip_eps=self.config["CLIP_EPS"],
            critic_coeff=self.config["CRITIC_COEFF"],
            entropy_coeff=self.config["ENTROPY_COEFF"],
        )

        train_state = train_state.apply_gradients(grads=grads)
        return (batch, targets, gae, train_state), total_loss

    def _calculate_gae(self, traj_batch):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = transition
            value_diff = self.config["GAMMA"] * next_value * (1 - done) - value
            delta = reward + value_diff
            gae = delta + self.config["GAMMA"] * self.config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae

        reverse_batch = (jnp.flip(traj_batch.done[:-1], axis=0), jnp.flip(traj_batch.value[:-1], axis=0), jnp.flip(traj_batch.reward[:-1], axis=0))
        last_value = traj_batch.value[-1]
        _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros(self.config["NUM_ENVS"]), last_value), reverse_batch)  # TODO: USE REVERSE INSTEAD?
        advantages = jnp.flip(advantages, axis=0)
        return advantages, advantages + traj_batch.value[:-1]