import osmnx as ox
import argparse
import networkx as nx
G = ox.graph_from_bbox(37.79, 37.78, -122.41, -122.43, network_type='drive')
nx.write_gpickle(G, "./simpleWorld.gpickle")
