from gym.envs.registration import register


register(
    id='simple-static-graph-v0',
    entry_point='gym_graph.envs:SimpleStaticEnv'
)

register(
    id='simple-random-graph-v0',
    entry_point='gym_graph.envs:SimpleRandomEnv'
)

register(
    id='hawaii-random-graph-v0',
    entry_point='gym_graph.envs:HawaiiRandomEnv'
)

register(
    id='hawaii-static-graph-v0',
    entry_point='gym_graph.envs:HawaiiStaticEnv'
)
