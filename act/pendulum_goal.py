import jax
import jax.numpy as jnp
from jax import random
import gym
import numpy as np
from typing import NamedTuple
from pendulum import Pendulum, PendulumState
from evo_utils import StaticVecNormalizer
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class PendulumGoalState(NamedTuple):
    pendulum_state: PendulumState
    goal_th: jnp.ndarray
    key: jnp.ndarray


class PendulumGoal:
    def __init__(self, config, network_apply=None, network_params=None, ob_mean=None, ob_var=None):
        self.env = Pendulum()
        self.config = config
        self.network_apply = network_apply
        self.network_params = network_params
        self.viewer = None
        self.ob_mean = ob_mean
        self.ob_var = ob_var
        self.epsilon = 1e-4

    def step(self, env_state, action):
        return self._step(env_state, action, self.network_params)

    def _step(self, env_state, action, network_params):
        if self.network_apply is None:
            mfos_obs = None
            pendulum_state, pendulum_obs, inner_reward, done, _ = self.env.step(env_state.pendulum_state, action)
        else:
            action = jnp.tanh(action) * self.config["CT_SCALE"]
            pendulum_ob = self.env.get_observation(env_state.pendulum_state)
            mfos_obs = jnp.concatenate([pendulum_ob, action], axis=-1)
            if self.ob_mean is not None and self.ob_var is not None:
                mfos_obs = (mfos_obs - self.ob_mean) / jnp.sqrt(self.ob_var + self.epsilon)
            v, pi = self.network_apply(network_params, mfos_obs)
            # pi = tfd.Categorical(logits=pi_out)
            action = pi.mode()
            pendulum_state, pendulum_obs, inner_reward, done, _ = self.env.step(env_state.pendulum_state, action)
            # raise NotImplementedError
        # state, _, _ = cartpole_state
        # x, x_dot, theta, theta_dot = state
        # reward = jnp.exp(-jnp.square(x - env_state.goal_x).sum())
        x = jnp.cos(env_state.pendulum_state.angle)
        y = jnp.sin(env_state.pendulum_state.angle)
        xy = jnp.concatenate([x, y])

        goal_x = jnp.cos(env_state.goal_th)
        goal_y = jnp.cos(env_state.goal_th)
        goal_xy = jnp.concatenate([goal_x, goal_y])

        # reward = -jnp.sum(jnp.square(xy - env_state.goal_xy), axis=-1)
        reward = jnp.exp(-1 * jnp.sum((xy - goal_xy) ** 2, axis=-1))

        env_state = PendulumGoalState(
            pendulum_state=pendulum_state,
            goal_th=env_state.goal_th,
            key=env_state.key,
        )
        env_state = self._maybe_reset(env_state, done)
        # return env_state, self.get_obs(env_state), reward, done, {"obs": pendulum_obs, "reward": inner_reward}
        return env_state, self.get_obs(env_state), reward, done, {"obs": pendulum_obs, "reward": inner_reward, "mfos_obs": mfos_obs, "action": action}

    def _maybe_reset(self, env_state, done):
        key = env_state.key
        return jax.lax.cond(done, self._reset, lambda pendulum_state, key: env_state, env_state.pendulum_state, key)

    def get_obs(self, env_state):
        obs = self.env.get_observation(env_state.pendulum_state)
        return jnp.concatenate([obs, jnp.cos(env_state.goal_th), jnp.sin(env_state.goal_th)], axis=-1)

    def get_obs_inner(self, env_state):
        obs = self.env.get_observation(env_state.pendulum_state)
        return obs

    def _reset(self, pendulum_state, key):
        key, goal_key_t = random.split(key, 2)
        # pendulum_state, pendulum_obs = self.env.reset(pendulum_key)

        goal_th = random.uniform(goal_key_t, minval=0, maxval=2.0 * jnp.pi, shape=(1,))

        env_state = PendulumGoalState(
            pendulum_state=pendulum_state,
            goal_th=goal_th,
            key=key,
        )
        return env_state

    def reset(self, key):
        key, pendulum_key = random.split(key, 2)
        pendulum_state, _ = self.env.reset(pendulum_key)
        env_state = self._reset(pendulum_state, key)
        # obs = jnp.concatenate([cartpole_obs, env_state.goal_x], axis=-1)
        return env_state, self.get_obs(env_state)

    def reset_zero(self, key):
        key, pendulum_key = random.split(key, 2)
        pendulum_state = self.env._reset(pendulum_key)
        env_state = self._reset(pendulum_state, key)
        # obs = jnp.concatenate([cartpole_obs, env_state.goal_x], axis=-1)
        return env_state, self.get_obs(env_state)

    def render(self, state, mode="human"):
        """
        Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        :param mode: Render mode, one of [human|rgb_array], default human
        :type mode: str, optional
        :return: Rendered image.
        """
        if self.viewer is None:
            # noinspection PyBroadException
            try:
                from gym.envs.classic_control import rendering
            except:
                if not self.env._logged_headless_message:
                    print("Unable to connect to display. Running the Ivy environment in headless mode...")
                    self.env._logged_headless_message = True
                return
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            # Pole.
            self.pole_geom = rendering.make_capsule(1, 0.2)
            self.pole_geom.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            self.pole_geom.add_attr(self.pole_transform)
            self.viewer.add_geom(self.pole_geom)

            # Goal Pole.
            self.goal_geom = rendering.make_capsule(1, 0.2)
            # self.goal_geom.set_color(0.8, 0.8, 0.8)
            self.goal_geom.set_color(0.9, 0.9, 0.2)
            self.goal_transform = rendering.Transform()
            self.goal_geom.add_attr(self.goal_transform)
            self.viewer.add_geom(self.goal_geom)

            # Axle.
            axle = rendering.make_circle(0.05)
            axle.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(axle)

        self.pole_transform.set_rotation(np.array(state.pendulum_state.angle)[0] + np.pi / 2)
        self.goal_transform.set_rotation(np.array(state.goal_th) + np.pi / 2)
        rew = np.array(self.env.get_reward(state.pendulum_state))
        self.pole_geom.set_color(1 - rew, rew, 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
