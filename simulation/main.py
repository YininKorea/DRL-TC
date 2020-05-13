import random, math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

n_topologies = 100
n_sensors = 19
scatter_area_radius = 1000 # meter
initial_energy = 1 # Joule
datasize = (500, 1000) # range of bits
e_p = 50e-9 # nJ/bit
rho = 1e-12 # pJ/m²/bit

#topologies = [generate_topology(nodes) for i in range(n_topologies)]
#lifetimes = [min([calculate_battery_lifetime(n) for n in topology.nodes]) for topology in topologies]

# visualize topologies (superimposed opaque)
# graph of 

def aggregate(node):
	return random.randint(datasize[0], datasize[1]) + sum([aggregate(child) for child in node.children])

def energy_consumption(node, bits, dist_to_parent):
	e_tx = rho * dist_to_parent**2
	return (e_p + e_tx) * bits

def dist(n1, n2):
	return ((n1.x-n2.x)**2 + (n1.y-n2.y)**2)**0.5

def calculate_battery_lifetime(node):
	initial_energy/energy_consumption(node, aggregate(node), dist(node, node.parent))

# generate topology --> star, random MST
def generate_topology(nodes):
	pass

def scatter_nodes(n_nodes, radius):
	nodes = []
	for i in range(n_nodes):
		a = random.random() * 360
		r = random.random() * radius
		nodes.append((i,{'coord':(math.cos(a)*r, math.sin(a)*r)}))
	return nodes

nodes = scatter_nodes(n_sensors, scatter_area_radius)
G = nx.Graph()
G.add_nodes_from(nodes)
nx.draw(G, pos=nodes)
plt.show()