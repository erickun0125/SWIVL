from gymnasium.envs.registration import register

register(
    id="gym_biart/BiArt-v0",
    entry_point="gym_biart.envs:BiArtEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state", "joint_type": "revolute"},
)
