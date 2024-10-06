import numpy as np


class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        z = next_obs[:, 0]
        done = (z < 1.0) + (z > 2.0)

        done = done[:, None]
        return done

    @staticmethod
    def reward_fn(obs, act, next_obs):
        reward_ctrl = -0.1 * np.sum(np.square(act), axis=1)
        reward_run = 0.25 / 0.015 * obs[:, 22]

        quad_impact_cost = .5e-6 * np.square(obs[:, -84:]).sum()
        quad_impact_cost = min(quad_impact_cost, 10)

        height = next_obs[:, 0]
        done = np.logical_or((height > 2.0), (height < 1.0))
        alive_reward = 5 * (1.0 - np.array(done, dtype=np.float))

        reward = reward_run + reward_ctrl + (-quad_impact_cost) + alive_reward
        return reward


