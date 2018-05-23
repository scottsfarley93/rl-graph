import osmnx
import argparse
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument("--place")
parser.add_argument("--out")
args = parser.parse_args()

print("Constructing graph from osm for ", args.place )
G = osmnx.graph_from_place(args.place, network_type="drive")

osmnx.plot_graph(osmnx.project_graph(G))

print("Saving new graph to ", args.out )
nx.write_gpickle(G, args.out)
