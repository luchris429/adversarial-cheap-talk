from jax import jit
from functools import partial
import jax.numpy as jnp
import jax
from typing import NamedTuple


class MfosStateStatic(NamedTuple):
    env_state: jnp.ndarray
    rng: jnp.ndarray


class MfosStateRNN(NamedTuple):
    env_state: jnp.ndarray
    mfos_state: jnp.ndarray
    rng: jnp.ndarray


class NormalizerState(NamedTuple):
    env_state: jnp.ndarray
    ob_mean: jnp.ndarray
    ob_var: jnp.ndarray
    count: float


class MfosWrapper:
    def __init__(self, env, network, mfos_params, config, init_mfos=None):
        self.env = env
        self.network = network
        self.mfos_params = mfos_params
        self.config = config
        if not self.config["STATIC"]:
            self.init_mfos = init_mfos

    def _reset(self, state, obs, rng):
        rng, rng_net = jax.random.split(rng, 2)

        if self.config["INCLUDE_ACTIONS"]:
            mfos_ob = jnp.concatenate([obs, jnp.zeros(self.config["NUM_ACTIONS"])], axis=-1)
        else:
            mfos_ob = obs

        if not self.config["STATIC"]:
            mfos_state, mfos_out = self.network({"params": self.mfos_params}, mfos_ob, self.init_mfos, rng_net)
            mstate = MfosStateRNN(
                env_state=state,
                mfos_state=mfos_state,
                rng=rng,
            )
        else:
            mfos_out = self.network({"params": self.mfos_params}, mfos_ob, rng_net)
            mstate = MfosStateStatic(
                env_state=state,
                rng=rng,
            )

        if self.config["GEN_EXPLOIT"]:
            obs = self.env.get_obs_inner(state)

        # THIS CODE WAS FOR THE ABLATIONS IN THE PAPER
        # if self.config["ADD_WRAPPER"]:
        #     if self.config["USELESS"]:
        #         mfos_out = jnp.concatenate([mfos_out, jnp.zeros_like(mfos_out)], axis=-1)
        #     obs = obs + mfos_out * self.config["CT_SCALE"]
        # else:
        obs = jnp.concatenate([obs, mfos_out * self.config["CT_SCALE"]], axis=-1)
        return mstate, obs

    def reset(self, rng):
        state, obs = self.env.reset(rng)
        return self._reset(state, obs, rng)

    def reset_zero(self, rng):
        state, obs = self.env.reset_zero(rng)
        return self._reset(state, obs, rng)

    def step(self, mstate, action):
        rstate, obs, reward, done, info = self.env.step(mstate.env_state, action)
        if self.config["INCLUDE_ACTIONS"]:
            if self.config["DISCRETE"]:
                action_onehot = jax.nn.one_hot(action, self.config["NUM_ACTIONS"])
                mfos_ob = jnp.concatenate([obs, action_onehot], axis=-1)
            else:
                mfos_ob = jnp.concatenate([obs, action], axis=-1)
        else:
            mfos_ob = obs

        rng, rng_net = jax.random.split(mstate.rng, 2)
        if not self.config["STATIC"]:
            mfos_state, mfos_out = self.network({"params": self.mfos_params}, mfos_ob, mstate.mfos_state, rng_net)
            mstate = MfosStateRNN(
                env_state=rstate,
                mfos_state=mfos_state,
                rng=rng,
            )
        else:
            mfos_out = self.network({"params": self.mfos_params}, mfos_ob, rng_net)
            mstate = MfosStateStatic(
                env_state=rstate,
                rng=rng,
            )

        if not self.config["GEN_EXPLOIT"]:
            # THIS CODE WAS FOR THE ABLATIONS IN THE PAPER
            # if self.config["ADD_WRAPPER"]:
            #     if self.config["USELESS"]:
            #         mfos_out = jnp.concatenate([mfos_out, jnp.zeros_like(mfos_out)], axis=-1)
            #     obs = obs + mfos_out * self.config["CT_SCALE"]
            # else:
            obs = jnp.concatenate([obs, mfos_out * self.config["CT_SCALE"]], axis=-1)
            return mstate, obs, reward, done, info
        else:
            assert not self.config["ADD_WRAPPER"]
            inner_obs, reward_inner = info["obs"], info["reward"]
            obs = jnp.concatenate([inner_obs, mfos_out * self.config["CT_SCALE"]], axis=-1)
            return mstate, obs, reward_inner, done, reward


class VecNormalizer:
    def __init__(self, env, config):
        self.env = env
        self.env_step = jax.vmap(env.step)
        self.env_reset = jax.vmap(env.reset)
        self.config = config
        self.epsilon = 1e-4

    def reset(self, rng):
        state, obs = self.env_reset(rng)
        vstate = NormalizerState(
            env_state=state,
            ob_mean=jnp.zeros(self.config["OBS_SIZE"]),
            ob_var=jnp.ones(self.config["OBS_SIZE"]),
            count=1e-4,
        )
        return vstate, obs

    def step(self, vstate, action):
        state, obs, reward, done, info = self.env_step(vstate.env_state, action)
        # UPDATE RMS
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = self.config["NUM_ENVS"] 

        delta = batch_mean - vstate.ob_mean
        tot_count = batch_count + vstate.count

        new_mean = vstate.ob_mean + delta * batch_count / tot_count
        m_a = vstate.ob_var * vstate.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * vstate.count * batch_count / tot_count
        new_var = M2 / tot_count

        vstate = NormalizerState(
            env_state=state,
            ob_mean=new_mean,
            ob_var=new_var,
            count=tot_count,
        )

        # APPLY RMS
        obs = (obs - vstate.ob_mean) / jnp.sqrt(vstate.ob_var + self.epsilon)

        return vstate, obs, reward, done, info

    def step_single(self, vstate, action):
        state, obs, reward, done, info = self.env.step(vstate.env_state, action)
        vstate = NormalizerState(
            env_state=state,
            ob_mean=vstate.ob_mean,
            ob_var=vstate.ob_var,
            count=vstate.count,
        )
        obs = (obs - vstate.ob_mean) / jnp.sqrt(vstate.ob_var + self.epsilon)

        return vstate, obs, reward, done, info


class StaticVecNormalizer:
    def __init__(self, env, config, ob_mean, ob_var):
        self.env = env
        self.config = config
        self.ob_mean = ob_mean
        self.ob_var = ob_var
        self.epsilon = 1e-4

    def reset(self, rng):
        state, obs = self.env.reset(rng)
        obs = (obs - self.ob_mean) / jnp.sqrt(self.ob_var + self.epsilon)
        return state, obs

    def reset_zero(self, rng):
        state, obs = self.env.reset_zero(rng)
        obs = (obs - self.ob_mean) / jnp.sqrt(self.ob_var + self.epsilon)
        return state, obs

    def step(self, state, action):
        state, obs, reward, done, info = self.env.step(state, action)
        obs = (obs - self.ob_mean) / jnp.sqrt(self.ob_var + self.epsilon)

        return state, obs, reward, done, info
