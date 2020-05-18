import random, math
import numpy as np

class Simulation():

	def __init__(self, n_nodes, scatter_area_radius=1000, initial_energy=1, datasize_range=(500, 1000), e_p=50e-9, rho=1e-12):
		self.n_nodes = n_nodes
		self.scatter_area_radius = scatter_area_radius
		self.initial_energy = initial_energy
		self.datasize_range = datasize_range
		self.e_p = e_p
		self.rho = rho
		self.node_positions = self.scatter_nodes(n_nodes, scatter_area_radius)

	def eval(self, topology):
		node_values = np.zeros(self.n_nodes)
		self.aggregate(0, node_values, topology)
		dist_to_parent = np.array([self.dist(self.node_positions[n], self.node_positions[np.argmax(topology[:,n])]) for n in range(self.n_nodes)])
		energy_consumption = (self.rho * dist_to_parent**2 + self.e_p) * node_values
		battery_lifetime = self.initial_energy / energy_consumption
		return battery_lifetime.min()

	def aggregate(self, node_idx, node_values, topology):
		children = np.nonzero(topology[node_idx])[0]
		data_size = random.randint(self.datasize_range[0], self.datasize_range[1]) + sum([self.aggregate(child, node_values, topology) for child in children])
		node_values[node_idx] = data_size
		return data_size

	def dist(self, n1, n2):
		return ((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2)**0.5

	def scatter_nodes(self, n_nodes, radius):
		nodes = []
		for i in range(n_nodes):
			a = random.random() * 360
			r = random.random() * radius
			nodes.append((math.cos(a)*r, math.sin(a)*r))
		return np.array(nodes)
