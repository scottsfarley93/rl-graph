#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_graph.envs.graph_env import GraphEnv

"""
A navigation environment on the island of Hawaii
The start and target nodes are the same every episode making learning easier

"""

import os
dirname = os.path.dirname(__file__)
graphFile = os.path.join(dirname, 'hi.gpickle')

class HawaiiRandomEnv(GraphEnv):
    def __init__(self):
        super().__init__(world=graphFile, isStatic=False)
