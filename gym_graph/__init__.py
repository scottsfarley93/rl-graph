from gym.envs.registration import register


register(
    id='graph-v0',
    entry_point='gym_graph.envs:GraphEnv'
)
