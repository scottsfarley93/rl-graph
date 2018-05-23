#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_graph.envs.graph_env import GraphEnv

"""
A navigation environment using a simple toy environment
The start and target nodes are regenerated every episode

"""

import os
dirname = os.path.dirname(__file__)
graphFile = os.path.join(dirname, 'simpleWorld.gpickle')
print(dirname, graphFile)
class SimpleRandomEnv(GraphEnv):
    def __init__(self):
        super().__init__(world=graphFile, isStatic=False, start_node=None, facing=None, target_node=None)
