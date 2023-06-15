"""Pendulum task adapted from:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""

# global
import jax
import jax.numpy as jnp
from jax import random
import gym
import numpy as np
from typing import NamedTuple


class PendulumState(NamedTuple):
    angle: jnp.ndarray
    angle_vel: jnp.ndarray
    key: jnp.ndarray
    t: int


# noinspection PyAttributeOutsideInit
class Pendulum(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):  # noqa
        """
        Initialize Pendulum environment
        """
        self.torque_scale = 1.0
        self.g = 9.8
        self.dt = 0.05
        self.m = 1.0
        self.l = 1.0
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=[1], dtype=np.float32)
        high = np.array([1.0, 1.0, np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.viewer = None
        self._logged_headless_message = False
        self.max_steps = 200
        self.kinematics_integrator = "euler"

    def get_observation(self, env_state):
        """
        Get observation from environment.
        :return: observation array
        """
        return jnp.concatenate((jnp.cos(env_state.angle), jnp.sin(env_state.angle), env_state.angle_vel), axis=-1)

    def get_reward(self, env_state):
        """
        Get reward based on current state
        :return: Reward array
        """
        # Pole verticality.
        rew = (jnp.cos(env_state.angle) + 1) / 2
        return rew[0]
        # return jnp.reshape(rew, (1,))

    # def get_state(self):
    #     """
    #     Get current state in environment.
    #     :return: angle and angular velocity arrays
    #     """
    #     return self.angle, self.angle_vel

    # def set_state(self, state):
    #     """
    #     Set current state in environment.
    #     :param state: tuple of angle and angular_velocity
    #     :type state: tuple of arrays
    #     :return: observation array
    #     """
    #     self.angle, self.angle_vel = state
    #     return self.get_observation()

    def _maybe_reset(self, env_state, done):
        key = env_state.key
        return jax.lax.cond(done, self._reset, lambda key: env_state, key)

    def _reset(self, key):
        key, key_ang, key_vel = random.split(key, 3)
        angle = random.uniform(key_ang, minval=-jnp.pi, maxval=jnp.pi, shape=(1,))
        angle_vel = random.uniform(key_vel, minval=-1.0, maxval=1.0, shape=(1,))
        env_state = PendulumState(
            angle=angle,
            angle_vel=angle_vel,
            key=key,
            t=0,
        )
        return env_state

    def reset(self, key):
        env_state = self._reset(key)
        key, key_t = random.split(env_state.key)
        t = random.randint(key_t, minval=0, maxval=self.max_steps, shape=())
        env_state = PendulumState(
            angle=env_state.angle,
            angle_vel=env_state.angle_vel,
            key=key,
            t=t,
        )
        return env_state, self.get_observation(env_state)

    def reset_zero(self, key):
        env_state = self._reset(key)
        return env_state, self.get_observation(env_state)

    def step(self, env_state, action):
        action = jnp.tanh(action)
        action = action * self.torque_scale

        angle_acc = -3 * self.g / (2 * self.l) * jnp.sin(env_state.angle + np.pi) + 3.0 / (self.m * self.l ** 2) * action
        if self.kinematics_integrator == "euler":
            angle_vel = env_state.angle_vel + self.dt * angle_acc
            angle = env_state.angle + self.dt * env_state.angle_vel
        else:  # semi-implicit euler
            angle_vel = env_state.angle_vel + self.dt * angle_acc
            angle = env_state.angle + self.dt * angle_vel

        env_state = PendulumState(
            angle=angle,
            angle_vel=angle_vel,
            key=env_state.key,
            t=env_state.t + 1,
        )
        done = env_state.t >= self.max_steps
        env_state = self._maybe_reset(env_state, done)

        return env_state, self.get_observation(env_state), self.get_reward(env_state), done, {}

    def render(self, env_state, mode="human"):
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
                if not self._logged_headless_message:
                    print("Unable to connect to display. Running the Ivy environment in headless mode...")
                    self._logged_headless_message = True
                return

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            # Pole.
            self.pole_geom = rendering.make_capsule(1, 0.2)
            self.pole_geom.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            self.pole_geom.add_attr(self.pole_transform)
            self.viewer.add_geom(self.pole_geom)

            # Axle.
            axle = rendering.make_circle(0.05)
            axle.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(axle)

        self.pole_transform.set_rotation(np.array(env_state.angle)[0] + np.pi / 2)
        rew = np.array(self.get_reward(env_state))
        self.pole_geom.set_color(1 - rew, rew, 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
