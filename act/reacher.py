"""Reacher task."""
import jax
import jax.numpy as jnp
from jax import random
import functools
import math
import gym
import numpy as np
from typing import NamedTuple


class ReacherState(NamedTuple):
    angles: jnp.ndarray
    angle_vels: jnp.ndarray
    goal_xy: jnp.ndarray
    key: jnp.ndarray
    t: int


class Reacher(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, num_joints=2):
        """
        Initialize Reacher environment
        :param num_joints: Number of joints in reacher.
        :type num_joints: int, optional
        """
        self.num_joints = num_joints
        self.torque_scale = 1.0
        # self.dt = 0.05
        self.dt = 0.1
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=[num_joints], dtype=np.float32)
        high = np.array([np.inf] * (num_joints * 3 + 2 + 2), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.viewer = None
        self._logged_headless_message = False
        self.max_steps = 200
        self.damping = 0.9
        self.kinematics_integrator = "euler"

    def get_observation(self, state):
        """
        Get observation from environment.

        :return: observation array
        """
        x = jnp.sum(jnp.cos(state.angles), axis=-1)
        y = jnp.sum(jnp.sin(state.angles), axis=-1)
        xy = jnp.stack([x, y])
        ob = jnp.concatenate(
            (
                jnp.cos(state.angles),
                jnp.sin(state.angles),
                state.angle_vels,
                state.goal_xy,
                state.goal_xy - xy,
            ),
            axis=0,
        )
        return ob

    def get_reward(self, state):
        """
        Get reward based on current state

        :return: Reward array
        """
        # Goal proximity.
        x = jnp.sum(jnp.cos(state.angles), axis=-1)
        y = jnp.sum(jnp.sin(state.angles), axis=-1)
        xy = jnp.stack([x, y])
        rew = jnp.exp(-1 * jnp.sum((xy - state.goal_xy) ** 2, axis=-1))
        return rew

    def get_reward_new(self, state, action):
        """
        Get reward based on current state

        :return: Reward array
        """
        # Goal proximity.
        x = jnp.sum(jnp.cos(state.angles), axis=-1)
        y = jnp.sum(jnp.sin(state.angles), axis=-1)
        xy = jnp.stack([x, y])
        # rew = -jnp.sqrt(jnp.sum((xy - state.goal_xy) ** 2, axis=-1)) - jnp.sum(jnp.square(action))
        rew = -jnp.sum(jnp.square(xy - state.goal_xy), axis=-1) - jnp.sum(jnp.square(action))

        return rew

    def get_done(self, state):
        x = jnp.sum(jnp.cos(state.angles), axis=-1)
        y = jnp.sum(jnp.sin(state.angles), axis=-1)
        xy = jnp.stack([x, y])
        dist = jnp.sqrt(jnp.sum((xy - state.goal_xy) ** 2, axis=-1))
        return jnp.logical_or(dist < 0.3, state.t >= self.max_steps)

    def _maybe_reset(self, env_state, done):
        key = env_state.key
        return jax.lax.cond(done, self._reset, lambda key: env_state, key)

    def _reset(self, key):
        key, key_angs, key_vels, key_r, key_t = random.split(key, 5)
        angles = random.uniform(key_angs, minval=-np.pi, maxval=np.pi, shape=(self.num_joints,))
        angle_vels = jnp.zeros(shape=(self.num_joints,))

        r = self.num_joints * jnp.sqrt(random.uniform(key_r, minval=0, maxval=1.0, shape=(1,)))
        theta = random.uniform(key_t, minval=0, maxval=2.0 * jnp.pi, shape=(1,))
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)

        env_state = ReacherState(
            angles=angles,
            angle_vels=angle_vels,
            goal_xy=jnp.concatenate([x, y], axis=0),
            key=key,
            t=0,
        )
        return env_state

    def reset(self, key):
        env_state = self._reset(key)
        key, key_t = random.split(env_state.key)
        t = random.randint(key_t, minval=0, maxval=self.max_steps, shape=())
        env_state = ReacherState(
            angles=env_state.angles,
            angle_vels=env_state.angle_vels,
            goal_xy=env_state.goal_xy,
            key=key,
            t=t,
        )
        return env_state, self.get_observation(env_state)

    def reset_zero(self, key):
        env_state = self._reset(key)
        return env_state, self.get_observation(env_state)

    def step(self, env_state, action):
        action = jnp.tanh(action)

        angle_accs = self.torque_scale * action
        if self.kinematics_integrator == "euler":
            angles = env_state.angles + self.dt * env_state.angle_vels
            angle_vels = env_state.angle_vels * self.damping + self.dt * angle_accs
        else:  # semi-implicit euler
            angle_vels = env_state.angle_vels * self.damping + self.dt * angle_accs
            angles = env_state.angles + self.dt * angle_vels
        env_state = ReacherState(angles=angles, angle_vels=angle_vels, goal_xy=env_state.goal_xy, key=env_state.key, t=env_state.t + 1)
        reward = self.get_reward_new(env_state, action)
        done = self.get_done(env_state)
        env_state = self._maybe_reset(env_state, done)
        return env_state, self.get_observation(env_state), reward, done, {}

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
                if not self._logged_headless_message:
                    print("Unable to connect to display. Running the Ivy environment in headless mode...")
                    self._logged_headless_message = True
                return
            self.viewer = rendering.Viewer(500, 500)
            bound = self.num_joints + 0.2
            self.viewer.set_bounds(-bound, bound, -bound, bound)

            # Goal.
            goal_geom = rendering.make_circle(0.2)
            goal_geom.set_color(0.4, 0.6, 1.0)
            self.goal_tr = rendering.Transform()
            goal_geom.add_attr(self.goal_tr)
            self.viewer.add_geom(goal_geom)

            # Arm segments and joints.
            l, r, t, b = 0, 1.0, 0.1, -0.1
            self.segment_trs = []
            for _ in range(self.num_joints):
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

        self.goal_tr.set_translation(*np.array(state.goal_xy).tolist())

        x, y = 0.0, 0.0
        for segment_tr, angle in zip(self.segment_trs, jnp.reshape(state.angles, (-1, 1))):
            segment_tr.set_rotation(np.array(angle)[0])
            segment_tr.set_translation(x, y)
            x = np.array(x + jnp.cos(jnp.expand_dims(angle, 0))[0])[0]
            y = np.array(y + jnp.sin(jnp.expand_dims(angle, 0))[0])[0]
        self.end_tr.set_translation(x, y)
        rew = np.array(self.get_reward(state))
        self.end_geom.set_color(1 - rew, rew, 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        """
        Close environment.
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
