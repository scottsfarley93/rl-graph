#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Simulate navigation along a road graph
Each episode is:
- a navigation from a random start point to a random end point
 - OR -
- a reward of [some negative score]
whichever comes first

The agent is presented with the following 7 actions:
 - Go straight
 - Make slight right turn
 - Make slight left turn
 - Make right turn
 - Make left turn
 - Make hard right turn
 - Make hard left turn
 - Make U-Turn

"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import math
import networkx
import matplotlib.pyplot as plt
import json
from math import sin, cos, radians, pi
from matplotlib import collections  as mc
import matplotlib.lines as mlines
import osmnx as ox
import pprint

HOPS = 5

print "imported libraries"

def loadGraph():
    G = ox.graph_from_bbox(37.79, 37.78, -122.41, -122.43, network_type='drive')
    for node in G.nodes:
        nodeAttr = G.node[node]
        G.node[node]['p'] = (nodeAttr['x'], nodeAttr['y'])
    return G



class GraphEnv(gym.Env):
    """
        Define a simple navigation environment.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__version__ = "0.1.0"
        print("Navigation Environment - Version {}".format(self.__version__))
        self.graph = self.getWorldGraph()

        self.current_episode = 0
        self.current_step = 0

        self.action_space = spaces.Discrete(7) ## seven available turn actions

        self.observation_space = spaces.Discrete(2) ## observations are an 8-length vector

        self.action_dict = {
            "STRAIGHT": 0,
            "SLIGHT_LEFT": 1,
            "LEFT": 2,
            "HARD_LEFT": 3,
            "U-TURN": 4,
            "HARD_RIGHT": 5,
            "RIGHT": 6,
            "SLIGHT_RIGHT": 7
        }

        self.actions_list = list(self.action_dict.keys())

    def getWorldGraph(self):
        ## TODO: the graph should come from the real world here
        simpleGraph = loadGraph()
        return simpleGraph

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        self.current_step += 1
        self._take_action(action)
        reward = self._get_reward()
        self.cum_reward += reward
        ob = self._get_state()
        done = self._is_episode_finished()
        return ob, reward, done, {}

    def reset(self):
        self.start_node = random.choice(list(self.graph.nodes))
        self.target_node = self.get_target_node(self.start_node, HOPS)
        self.current_episode += 1
        self.current_step = 0
        self.facing = random.choice(range(0, 360)) ## random start angle
        self.current_node = self.start_node
        self.start_position = self.graph.node[self.start_node]['p']
        self.end_position = self.graph.node[self.target_node]['p']
        self.neighbors = self.getConnections(self.start_node)
        self.x = self.start_position[0]
        self.y = self.start_position[1]
        self.actions = self.getActions()
        self.history = []
        self.last_distance_travelled = 0
        self.action_blocked = False
        self.cum_reward = 0

        self.distance_traveled = 0
        self.geo_distance_to_goal = self.getGeoDistance(self.start_position, self.end_position)
        # print "reset environment to:"
        # print "\tStart Node: ", self.start_node, "(", self.start_position, ")"
        # print "\tEnd Node: ", self.target_node, "(", self.end_position, ")"
        # print "\tStart Direction: ", self.facing
        return self._get_state()

    def get_target_node(self, start_node, hops):
        current_node = start_node
        for i in range(0, hops):
            choices = list(self.graph.neighbors(current_node))
            if (len(choices) == 0):
                continue
            next_node = random.choice(choices)
            current_node = next_node
        return current_node

    def render(self, mode='human', close=False, labels=False):
        fig = plt.figure(figsize=(10, 10))
        nodes = list(self.graph.nodes)
        pts = []
        labs = []
        for n in nodes:
            nodeInfo = self.graph.node[n]
            pts.append(nodeInfo['p'])
            labs.append(n)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        ## labels
        if False:
            for i in range(0, len(labs)):
                fig.text(xs[i], ys[i], labs[i])

        plt.title("Iteration: " + str(self.current_step) + " (epoch=" + str(self.current_episode) + ")")

        ## path history
        for segment in self.history:
            ax = fig.gca()
            _x = (segment[0][0], segment[1][0])
            _y = (segment[0][1], segment[1][1])

            l = mlines.Line2D(_x, _y)

            ax.add_line(l)

        ## connected paths
        for node in self.graph.nodes:

            _source = self.graph.node[node]['p']
            _targets = self.graph.neighbors(node)
            for t in _targets:
                _target = self.graph.node[t]['p']

                ax = fig.gca()
                _x = (_source[0], _target[0])
                _y = (_source[1], _target[1])

                l = mlines.Line2D(_x, _y, color='gray')
                ax.add_line(l)

        ## node points
        plt.scatter(xs, ys, color='gray')

        ## direction arrow
        x1, y1 = self.angleToPos(self.facing, self.x, self.y, d=0.001)
        ax = fig.gca()
        _x = (self.x, x1)
        _y = (self.y, y1)

        l = mlines.Line2D(_x, _y, color='red')
        ax.add_line(l)
        plt.plot([self.end_position[0]], [self.end_position[1]], 'ro',  color='blue',markersize=12, zorder=10)



    def angleToPos(self, theta, x0, y0, d=0.001):
        theta_rad = radians(theta)
        x1 = (x0 - d*cos(theta_rad))
        y1 = (y0 - d*sin(theta_rad))
        return x1, y1

    def getNodeXY(self, node):
        return self.graph.node[node]["p"]

    def getAngleToNextNode(self, nodeA, nodeB):
        posA = self.getNodeXY(nodeA)
        posB = self.getNodeXY(nodeB)
        lat1 = math.radians(posA[0])
        lat2 = math.radians(posB[0])

        diffLong = math.radians(posB[1] - posA[1])

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                * math.cos(lat2) * math.cos(diffLong))

        initial_bearing = math.atan2(x, y)
        degs = round(math.degrees(initial_bearing), 1)
        return degs

    def getAnglesToNeighbors(self, node, referenceAngle):
        angles = []
        neighbors = self.graph.neighbors(node)
#         pxs = []
#         pys = []
#         for node in neighbors:
#             p = self.graph.node[node]['p']
#             pxs.append(p[0])
#             pys.append(p[1])
#         plt.scatter(pxs, pys, color='green')
#         current = self.graph.node[node]['p']
#         plt.scatter([current[0]], [current[1]], color='red')
#         plt.show()
        for n in neighbors:
            d = self.getAngleToNextNode(node, n) - referenceAngle
            if d < -180:
                d = d + 360
            if d > 180:
                d = d - 360
            angles.append(d)
        return angles

    def actionFromAngle(self, angle):
        if angle == 0:
            return "STRAIGHT"
        elif angle > 0 and angle < 60:
            return "SLIGHT_LEFT"
        elif angle >=45 and angle < 120:
            return "LEFT"
        elif angle >= 120 and angle < 180:
            return "HARD_LEFT"
        elif angle == 180 or angle == -180:
            return "U_TURN"
        elif angle > -60 and angle < 0:
            return "SLIGHT_RIGHT"
        elif angle <= -60 and angle > -120:
            return "RIGHT"
        elif angle <= -120 and angle > -180:
            return "HARD_RIGHT"
        else:
            raise ValueError("invalid angle")

    def getReferenceAngle(self, oldNode, newNode):
        return getAngleToNextNode(oldNode, newNode)

    def getConnections(self, node):
        self.graph[node].keys()
        return list(self.graph.neighbors(node))

    def getGeoDistance(self, node1, node2):
        return math.hypot(node2[0] - node2[0], node2[1] - node1[1])

    def _take_action(self,action):
        self.actions = self.getActions()
        thisAction = self.actions[action]
        destination = thisAction['destination']
        angle = thisAction['angle']

        if destination is None: ## there are no connections given this action
            # print "\t --> Action blocked."
            self.action_blocked = True
            return ## don't do anything
        else: ## connection is possible, go there
            # print "Took Action: ", self.actions_list[action]
            self.action_blocked = False
            self.move_to_node(destination, angle)

    def _get_reward(self):
        if self.current_node == self.target_node:
            return 1000
        if self.action_blocked:
            return -25
        else:
            return 0

    def _get_state(self):
        ## state obsveration space at node N
        ## - total geographic distance traveled from start_node
        ## - total navigated distance travlled from start_node
        ## - geographic distance remaining to target_node
        ## - angle to target_node
        ## - number of neighbors
        ## - for each possible action:
        ##  - distance to next node
        ##  - angle to next node
        ##  - highway class of edge to next node
        ##  - edge to next node is oneway
        ## later:
        ## - time of day
        ## - hourly traffic volume
        angle_to_goal = self.getAngleToNextNode(self.current_node, self.target_node)
        geoDistanceTravelled = self.getGeoDistance(self.start_position, [self.x, self.y])
        navigatedDistance = self.distance_traveled
        distanceToGoal = self.geo_distance_to_goal
        numNeighbors = len(list(self.graph.neighbors(self.current_node)))
        nextAttrs = []
        actions = self.getActions()
        for action in actions:
            destination = action['destination']
            action_obs = self.get_edge_obs(self.current_node, destination)
            angle = action['angle']
            if angle is None:
                angle = -9999
            action_obs.append(angle)
            nextAttrs = nextAttrs + action_obs
        return [angle_to_goal, geoDistanceTravelled, navigatedDistance, distanceToGoal, numNeighbors] + nextAttrs

    def get_edge_obs(self, nodeA, nodeB):
        ##  - distance to next node
        ##  - angle to next node
        ##  - highway class of edge to next node
        ##  - edge to next node is oneway
        edge = self.graph.get_edge_data(nodeA, nodeB, 0)
        attrs = []
        if edge is not None:
            try:
                distance = edge['length'];
            except:
                distance = -9999
            try:
                highway = self.highwayClass2Number(edge['highway']);
            except:
                highway = -9999
            # try:
            #     oneway = int(edge['oneway']);
            # except:
            #     oneway = False
            attrs = attrs + [distance, highway]
        else:
            attrs = attrs + [-9999, -9999]
        return attrs


    def _is_episode_finished(self):
        if  self.current_node == self.target_node:
            print "reached goal!"
            return True
        elif self.cum_reward < -10000:
            return True
        else:
            return False
    def move_to_node(self, node, facing):
        _node = self.current_node
        self.current_node = node
        geoPos = self.graph.node[node]['p']
        self.x = geoPos[0]
        self.y = geoPos[1]


        self.facing = facing
        self.neighbors = self.getConnections(self.current_node)
        self.actions = self.getActions()

        _lastXY = self.graph.node[_node]['p']
        _lastDistance = self.getGeoDistance(_lastXY, geoPos)
        self.last_distance_travelled = _lastDistance

        self.history.append([_lastXY, geoPos])
        self.distance_traveled += _lastDistance
        self.geo_distance_to_goal = self.getGeoDistance(geoPos, self.end_position)
        # print "Moved to node: ", self.current_node
        # print "\tX=", self.x
        # print "\tY=", self.y
        # print "\tFacing ", self.facing
        return

    def highwayClass2Number(self, classification):
        if classification == "motorway":
            return 75
        elif classification == "trunk":
            return 60
        elif classification == "primary":
            return 50
        elif classification == "secondary":
            return 45
        elif classification == "tertiary":
            return 25
        elif classification == "residential":
            return 25
        else:
            return 5

    def getActions(self):
        _neighbors = self.getConnections(self.current_node)
        _connectionAngles = self.getAnglesToNeighbors(self.current_node, self.facing)
        _allowed_actions = [] ## reset
        for angle in _connectionAngles:
            action = self.actionFromAngle(int(angle)) ## okay
            _allowed_actions.append(action)

        _actions = []

        idx = 0
        for action in self.actions_list: ## iterate through all the actions the agent knows how to do
            ## TODO: calculate cost here for reward
            if action in _allowed_actions:
                result = {"destination" : _neighbors[idx], "angle": _connectionAngles[idx], "action": action}
                idx += 1
            else:
                result = {"destination": None, "angle": None, "action": action}
            _actions.append(result)
        return _actions
