import jax
import jax.numpy as jnp
from jax import random
import gym
import numpy as np
from typing import NamedTuple
from cartpole import JaxCartPole
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class CartPoleGoalState(NamedTuple):
    cartpole_state: jnp.ndarray
    goal_x: jnp.ndarray
    key: jnp.ndarray


class JaxCartPoleGoal:
    """
    V1: HIT A SPECIFIED TARGET X POSITION
    """

    def __init__(self, config, network_apply=None, network_params=None, ob_mean=None, ob_var=None):
        self.env = JaxCartPole()
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
            cartpole_state, cartpole_obs, inner_reward, done, _ = self.env.step(env_state.cartpole_state, action)
        else:
            action = jnp.tanh(action) * self.config["CT_SCALE"]
            mfos_obs = jnp.concatenate([env_state.cartpole_state[0], action], axis=-1)
            if self.ob_mean is not None and self.ob_var is not None:
                mfos_obs = (mfos_obs - self.ob_mean) / jnp.sqrt(self.ob_var + self.epsilon)
            v, pi = self.network_apply(network_params, mfos_obs)
            # pi = tfd.Categorical(logits=pi_out)
            action = pi.mode()
            cartpole_state, cartpole_obs, inner_reward, done, _ = self.env.step(env_state.cartpole_state, action)
            # raise NotImplementedError
        state, _, _ = cartpole_state
        x, x_dot, theta, theta_dot = state
        reward = jnp.exp(-jnp.square(x - env_state.goal_x).sum())
        env_state = CartPoleGoalState(
            cartpole_state=cartpole_state,
            goal_x=env_state.goal_x,
            key=env_state.key,
        )
        env_state = self._maybe_reset(env_state, done)
        return env_state, self.get_obs(env_state), reward, done, {"obs": cartpole_obs, "reward": inner_reward, "mfos_obs": mfos_obs, "action": action}

    def _maybe_reset(self, env_state, done):
        key = env_state.key
        return jax.lax.cond(done, self._reset, lambda cartpole_state, key: env_state, env_state.cartpole_state, key)

    def get_obs(self, env_state):
        return jnp.concatenate([env_state.cartpole_state[0], env_state.goal_x], axis=-1)

    def get_obs_inner(self, env_state):
        return env_state.cartpole_state[0]

    def _reset(self, cartpole_state, key):
        key, goal_key = random.split(key, 2)
        # cartpole_state, cartpole_obs = self.env.reset(cartpole_key)
        goal_x = random.uniform(goal_key, minval=-self.env.x_threshold, maxval=self.env.x_threshold, shape=(1,))
        env_state = CartPoleGoalState(
            cartpole_state=cartpole_state,
            goal_x=goal_x,
            key=key,
        )
        return env_state

    def reset(self, key):
        key, cartpole_key = random.split(key, 2)
        cartpole_state, _ = self.env.reset(cartpole_key)
        env_state = self._reset(cartpole_state, key)
        # obs = jnp.concatenate([cartpole_obs, env_state.goal_x], axis=-1)
        return env_state, self.get_obs(env_state)

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

        state, t, key = env_state.cartpole_state
        x, x_dot, theta, theta_dot = state
        screen_width = 500
        screen_height = 500
        world_width = 4
        scale = screen_width / world_width
        pole_width = 10.0
        pole_len = scale * (2 * self.env.length)
        cart_width = 50.0
        cart_height = 30.0
        cart_y = screen_height / 2

        if self.viewer is None:
            # noinspection PyBroadException
            try:
                from gym.envs.classic_control import rendering
            except:
                if not self.env._logged_headless_message:
                    print("Unable to connect to display. Running the Ivy environment in headless mode...")
                    self.env._logged_headless_message = True
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

            # Goal Cart.
            l = -cart_width / 2
            r = cart_width / 2
            t = cart_height / 2
            b = -cart_height / 2
            goal_geom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.goal_tr = rendering.Transform()
            goal_geom.add_attr(self.goal_tr)
            # goal_geom.set_color(0.8, 0.8, 0.8)
            goal_geom.set_color(0.9, 0.9, 0.2)
            self.viewer.add_geom(goal_geom)

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
        goal_x = np.array(env_state.goal_x * scale + screen_width / 2.0)
        self.cart_tr.set_translation(cart_x, cart_y)
        self.goal_tr.set_translation(goal_x, cart_y)
        self.pole_tr.set_rotation(-np.array(theta))
        # rew = np.array(self.get_reward(env_state))[0]
        # rew = np.array(jnp.exp(-jnp.square(x - env_state.goal_x).sum()))
        rew = 1.0
        self.pole_geom.set_color(1 - rew, rew, 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
