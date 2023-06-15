import jax
import jax.numpy as jnp
from jax import random
import gym
import numpy as np
from typing import NamedTuple
from reacher import Reacher, ReacherState
from evo_utils import StaticVecNormalizer
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class ReacherGoalState(NamedTuple):
    reacher_state: ReacherState
    goal_xy: jnp.ndarray
    key: jnp.ndarray


class ReacherGoal:
    def __init__(self, config, network_apply=None, network_params=None, ob_mean=None, ob_var=None):
        self.env = Reacher()
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
        if network_params is None:
            mfos_obs = None
            reacher_state, reacher_obs, inner_reward, done, _ = self.env.step(env_state.reacher_state, action)
        else:
            action = jnp.tanh(action) * self.config["CT_SCALE"]
            reacher_ob = self.env.get_observation(env_state.reacher_state)
            mfos_obs = jnp.concatenate([reacher_ob, action], axis=-1)
            if self.ob_mean is not None and self.ob_var is not None:
                mfos_obs = (mfos_obs - self.ob_mean) / jnp.sqrt(self.ob_var + self.epsilon)
            v, pi = self.network_apply(network_params, mfos_obs)
            # pi = tfd.Categorical(logits=pi_out)
            action = pi.mode()
            reacher_state, reacher_obs, inner_reward, done, _ = self.env.step(env_state.reacher_state, action)
            # raise NotImplementedError
        # state, _, _ = cartpole_state
        # x, x_dot, theta, theta_dot = state
        # reward = jnp.exp(-jnp.square(x - env_state.goal_x).sum())
        x = jnp.sum(jnp.cos(env_state.reacher_state.angles), axis=-1)
        y = jnp.sum(jnp.sin(env_state.reacher_state.angles), axis=-1)
        xy = jnp.stack([x, y])

        # reward = -jnp.sum(jnp.square(xy - env_state.goal_xy), axis=-1)
        reward = jnp.exp(-1 * jnp.sum((xy - env_state.goal_xy) ** 2, axis=-1))

        env_state = ReacherGoalState(
            reacher_state=reacher_state,
            goal_xy=env_state.goal_xy,
            key=env_state.key,
        )
        env_state = self._maybe_reset(env_state, done)
        # return env_state, self.get_obs(env_state), reward, done, {"obs": reacher_obs, "reward": inner_reward}
        return env_state, self.get_obs(env_state), reward, done, {"obs": reacher_obs, "reward": inner_reward, "mfos_obs": mfos_obs, "action": action}

    def _maybe_reset(self, env_state, done):
        key = env_state.key
        return jax.lax.cond(done, self._reset, lambda reacher_state, key: env_state, env_state.reacher_state, key)

    def get_obs(self, env_state):
        obs = self.env.get_observation(env_state.reacher_state)
        return jnp.concatenate([obs, env_state.goal_xy], axis=-1)

    def get_obs_inner(self, env_state):
        obs = self.env.get_observation(env_state.reacher_state)
        return obs

    def _reset(self, reacher_state, key):
        key, goal_key_t, goal_key_r = random.split(key, 3)
        # reacher_state, reacher_obs = self.env.reset(reacher_key)

        r = self.env.num_joints * jnp.sqrt(random.uniform(goal_key_r, minval=0, maxval=1.0, shape=(1,)))
        theta = random.uniform(goal_key_t, minval=0, maxval=2.0 * jnp.pi, shape=(1,))
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        goal_xy = jnp.concatenate([x, y], axis=0)

        # goal_xy = random.uniform(goal_key, minval=-self.env.x_threshold, maxval=self.env.x_threshold, shape=(1,))

        env_state = ReacherGoalState(
            reacher_state=reacher_state,
            goal_xy=goal_xy,
            key=key,
        )
        return env_state

    def reset(self, key):
        key, reacher_key = random.split(key)
        reacher_state, _ = self.env.reset(reacher_key)
        env_state = self._reset(reacher_state, key)
        # obs = jnp.concatenate([cartpole_obs, env_state.goal_x], axis=-1)
        return env_state, self.get_obs(env_state)

    def reset_zero(self, key):
        key, reacher_key = random.split(key)
        reacher_state = self.env._reset(reacher_key)
        env_state = self._reset(reacher_state, key)
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
            bound = self.env.num_joints + 0.2
            self.viewer.set_bounds(-bound, bound, -bound, bound)

            # Goal.
            goal_geom = rendering.make_circle(0.2)
            goal_geom.set_color(0.4, 0.6, 1.0)
            self.goal_tr = rendering.Transform()
            goal_geom.add_attr(self.goal_tr)
            self.viewer.add_geom(goal_geom)

            # Fake.
            fake_geom = rendering.make_circle(0.2)
            # fake_geom.set_color(1.0, 0.4, 0.6)
            fake_geom.set_color(0.9, 0.9, 0.2)
            self.fake_tr = rendering.Transform()
            fake_geom.add_attr(self.fake_tr)
            self.viewer.add_geom(fake_geom)

            # Arm segments and joints.
            l, r, t, b = 0, 1.0, 0.1, -0.1
            self.segment_trs = []
            for _ in range(self.env.num_joints):
                # Segment.
                segment_geom = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                segment_tr = rendering.Transform()
                self.segment_trs.append(segment_tr)
                segment_geom.add_attr(segment_tr)
                segment_geom.set_color(0.0, 0.0, 0.0)
                self.viewer.add_geom(segment_geom)

                # Joint.
                joint_geom = rendering.make_circle(0.1)
                joint_geom.set_color(0.5, 0.5, 0.5)
                joint_geom.add_attr(segment_tr)
                self.viewer.add_geom(joint_geom)

            # End effector.
            self.end_geom = rendering.make_circle(0.1)
            self.end_tr = rendering.Transform()
            self.end_geom.add_attr(self.end_tr)
            self.viewer.add_geom(self.end_geom)

        self.goal_tr.set_translation(*np.array(state.reacher_state.goal_xy).tolist())
        self.fake_tr.set_translation(*np.array(state.goal_xy).tolist())

        x, y = 0.0, 0.0
        for segment_tr, angle in zip(self.segment_trs, jnp.reshape(state.reacher_state.angles, (-1, 1))):
            segment_tr.set_rotation(np.array(angle)[0])
            segment_tr.set_translation(x, y)
            x = np.array(x + jnp.cos(jnp.expand_dims(angle, 0))[0])[0]
            y = np.array(y + jnp.sin(jnp.expand_dims(angle, 0))[0])[0]
        self.end_tr.set_translation(x, y)
        rew = np.array(self.env.get_reward(state.reacher_state))
        self.end_geom.set_color(1 - rew, rew, 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
