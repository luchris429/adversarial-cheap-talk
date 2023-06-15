import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Any

class GymnaxWrapper:
    def __init__(self, env, env_params):
        self.env = env
        self.env_params = env_params

        # HACKY
        self.observation_space = np.zeros([np.prod(self.env.observation_space(self.env_params).shape), 1])

    def reset(self, key_reset):
        key_env, key_reset = jax.random.split(key_reset)
        obs, state = self.env.reset(key_reset, self.env_params)
        env_state = (state, key_env)
        obs = jnp.ravel(obs)
        return env_state, obs

    def step(self, env_state, action):
        state, key_env = env_state
        key_env, key_step = jax.random.split(key_env)
        obs, state, reward, done, _ = self.env.step(key_step, state, action, self.env_params)
        obs = jnp.ravel(obs)
        env_state = (state, key_env)
        env_state, obs = self._maybe_reset((env_state, obs), done)
        return env_state, obs, reward, done, {}
    
    def get_obs(self, env_state):
        obs = self.env.get_obs(env_state[0])
        obs = jnp.ravel(obs)
        return obs

    def _maybe_reset(self, env_state_and_obs, done):
        key = env_state_and_obs[0][1]
        return jax.lax.cond(done, self.reset, lambda key: env_state_and_obs, key)

class GymnaxGoalState(NamedTuple):
    gymnax_state: Any
    goal_return: jnp.float32
    cum_r: jnp.float32
    key: jnp.float32

class GymnaxGoalWrapper:
    def __init__(self, env, config, network_apply=None, network_params=None, ob_mean=None, ob_var=None):
        self.env = env
        self.config = config
        self.network_apply = network_apply
        self.network_params = network_params
        self.viewer = None
        self.reset_zero = self.reset
        self.ob_mean = ob_mean
        self.ob_var = ob_var
        self.epsilon = 1e-4

    def step(self, env_state, action):
        return self._step(env_state, action, self.network_params)

    def _step(self, env_state, action, network_params):
        if self.network_apply is None:
            mfos_obs = None
            gymnax_state, inner_obs, inner_reward, done, _ = self.env.step(env_state.gymnax_state, action)
        else:
            action = jnp.tanh(action) * self.config["CT_SCALE"]
            gymnax_ob = self.env.get_obs(env_state.gymnax_state)
            mfos_obs = jnp.concatenate([gymnax_ob, action], axis=-1)
            if self.ob_mean is not None and self.ob_var is not None:
                mfos_obs = (mfos_obs - self.ob_mean) / jnp.sqrt(self.ob_var + self.epsilon)
            v, pi = self.network_apply(network_params, mfos_obs)
            action = pi.mode()
            gymnax_state, gymnax_obs, inner_reward, done, _ = self.env.step(env_state.gymnax_state, action)

        reward = (env_state.cum_r <= env_state.goal_return) * inner_reward - (env_state.cum_r > env_state.goal_return) * inner_reward
        reward = reward[0]
        cum_r = env_state.cum_r + inner_reward
        env_state = GymnaxGoalState(
            gymnax_state=gymnax_state,
            goal_return=env_state.goal_return,
            cum_r=cum_r,
            key=env_state.key,
        )
        env_state = self._maybe_reset(env_state, done)
        return env_state, self.get_obs(env_state), reward, done, {"obs": gymnax_obs, "reward": inner_reward, "mfos_obs": mfos_obs, "action": action}

    def _maybe_reset(self, env_state, done):
        key = env_state.key
        return jax.lax.cond(done, self._reset, lambda gymnax_state, key: env_state, env_state.gymnax_state, key)

    def get_obs(self, env_state):
        obs = self.env.get_obs(env_state.gymnax_state)
        return jnp.concatenate([obs, env_state.goal_return, env_state.cum_r], axis=-1)

    def get_obs_inner(self, env_state):
        return self.env.get_obs(env_state.gymnax_state)

    def _reset(self, gymnax_state, key):
        key, goal_key = jax.random.split(key, 2)
        # goal_x = random.uniform(goal_key, minval=-self.env.x_threshold, maxval=self.env.x_threshold, shape=(1,))
        goal_return = jax.random.uniform(goal_key, minval=0.0, maxval=50.0, shape=(1,))
        env_state = GymnaxGoalState(
            gymnax_state=gymnax_state,
            goal_return=goal_return,
            cum_r=jnp.zeros((1,)),
            key=key,
        )
        return env_state

    def reset(self, key):
        key, gymnax_key = jax.random.split(key, 2)
        gymnax_state, _ = self.env.reset(gymnax_key)
        env_state = self._reset(gymnax_state, key)
        # obs = jnp.concatenate([cartpole_obs, env_state.goal_x], axis=-1)
        return env_state, self.get_obs(env_state)