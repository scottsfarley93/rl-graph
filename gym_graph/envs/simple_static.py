#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_graph.envs.graph_env import GraphEnv

"""
A navigation environment using a simple toy environment
The start and target nodes remain static

"""

import os
dirname = os.path.dirname(__file__)
graphFile = os.path.join(dirname, 'simpleWorld.gpickle')

print(dirname, graphFile)

class SimpleStaticEnv(GraphEnv):
    def __init__(self):
        super().__init__(world=graphFile, isStatic=True, start_node=2351399563, facing=0, target_node=342548255)
