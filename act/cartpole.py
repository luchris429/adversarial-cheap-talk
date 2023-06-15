"""
INSPIRED BY:
https://medium.com/@ngoodger_7766/writing-an-rl-environment-in-jax-9f74338898ba
"""
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import functools
import math
from gym import spaces


class JaxCartPole:
    """
    Based on OpenAI Gym Cartpole
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.viewer = None

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.random_limit = 0.05

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.max_tstep = 200

    # @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, env_state, action):
        state, t, key = env_state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * (2 * action - 1)
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        done = (x < -self.x_threshold) | (x > self.x_threshold) | (theta > self.theta_threshold_radians) | (theta < -self.theta_threshold_radians) | (t >= self.max_tstep)
        reward = 1.0

        env_state = jnp.array([x, x_dot, theta, theta_dot]), t + 1, key
        env_state = self._maybe_reset(env_state, done)
        new_state = env_state[0]
        return env_state, self._get_obsv(new_state), reward, done, {}

    def _get_obsv(self, state):
        return state

    def _maybe_reset(self, env_state, done):
        key = env_state[2]
        return jax.lax.cond(done, self._reset, lambda key: env_state, key)

    def _reset(self, key):
        new_state = random.uniform(key, minval=-self.random_limit, maxval=self.random_limit, shape=(4,))
        new_key = random.split(key)[0]
        return new_state, 0, new_key

    # @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        env_state = self._reset(key)
        initial_state = env_state[0]
        return env_state, self._get_obsv(initial_state)

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
        state, t, key = env_state
        x, x_dot, theta, theta_dot = state
        screen_width = 500
        screen_height = 500
        world_width = 4
        scale = screen_width / world_width
        pole_width = 10.0
        pole_len = scale * (2 * self.length)
        cart_width = 50.0
        cart_height = 30.0
        cart_y = screen_height / 2

        if self.viewer is None:
            # noinspection PyBroadException
            try:
                from gym.envs.classic_control import rendering
            except:
                if not self._logged_headless_message:
                    print("Unable to connect to display. Running the Ivy environment in headless mode...")
                    self._logged_headless_message = True
                return

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Track.
            track = rendering.Line((0.0, cart_y), (screen_width, cart_y))
            track.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(track)

            # Cart.
            l = -cart_width / 2
            r = cart_width / 2
            t = cart_height / 2
            b = -cart_height / 2
            cart_geom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_tr = rendering.Transform()
            cart_geom.add_attr(self.cart_tr)
            cart_geom.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(cart_geom)

            # Pole.
            l = -pole_width / 2
            r = pole_width / 2
            t = pole_len - pole_width / 2
            b = -pole_width / 2
            self.pole_geom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.pole_tr = rendering.Transform(translation=(0, 0))
            self.pole_geom.add_attr(self.pole_tr)
            self.pole_geom.add_attr(self.cart_tr)
            self.viewer.add_geom(self.pole_geom)

            # Axle.
            axle_geom = rendering.make_circle(pole_width / 2)
            axle_geom.add_attr(self.pole_tr)
            axle_geom.add_attr(self.cart_tr)
            axle_geom.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(axle_geom)

        cart_x = np.array(x * scale + screen_width / 2.0)
        self.cart_tr.set_translation(cart_x, cart_y)
        self.pole_tr.set_rotation(-np.array(theta))
        # rew = np.array(self.get_reward(env_state))[0]
        rew = 1
        self.pole_geom.set_color(1 - rew, rew, 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
