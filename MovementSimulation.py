import random, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

class Simulation():

	def __init__(self, n_nodes, scatter_area_radius=1000, initial_energy=1, datasize_range=(500, 1000), e_p=50e-9, rho=1e-12, seed=None):
		random.seed(seed)
		np.random.seed(seed)
		self.direction_delta = 10
		self.n_nodes = n_nodes
		self.scatter_area_radius = scatter_area_radius
		self.initial_energy = initial_energy
		self.datasize_range = datasize_range
		self.e_p = e_p
		self.rho = rho
		self.node_positions = self.scatter_nodes(n_nodes, scatter_area_radius)
		self.node_distances = euclidean_distances(self.node_positions)
		self.directions = self.init_movement(n_nodes)
		self.node_datasizes = np.random.randint(self.datasize_range[0], self.datasize_range[1], n_nodes)

	def eval(self, topology):
		positions = self.node_positions.copy()
		directions = self.directions.copy()
		battery = np.full((self.n_nodes), self.initial_energy, dtype='float')
		lifetime = 0
		node_values = np.zeros(self.n_nodes)
		self.aggregate(0, node_values, topology)
		while not (battery <= 0).any():
			dist_to_parent = euclidean_distances(positions)[range(self.n_nodes), np.argmax(topology, axis=0)]
			energy_consumption = (self.rho * dist_to_parent**2 + self.e_p) * node_values
			battery -= energy_consumption
			battery[0] = 1 # make sure gateway is not the minimum (has theoretical infinite lifetime)
			lifetime += 1
			positions += directions
			directions[np.linalg.norm(positions) >= self.scatter_area_radius] *= -1
			directions = np.random.normal(size=(self.n_nodes, 2))+directions
			directions = directions/np.linalg.norm(directions)
		return lifetime

	def aggregate(self, node_idx, node_values, topology):
		children = np.nonzero(topology[node_idx])[0]
		data_size = self.node_datasizes[node_idx]
		for child in children:
			data_size += self.aggregate(child, node_values, topology)
		node_values[node_idx] = data_size
		return data_size

	def scatter_nodes(self, n_nodes, radius):
		nodes = []
		for i in range(n_nodes):
			a = random.random() * 360
			r = random.random() * radius
			nodes.append([math.cos(a)*r, math.sin(a)*r])
		return np.array(nodes)

	def init_movement(self, n_nodes):
		vectors = np.random.random((n_nodes, 2))
		return vectors/np.linalg.norm(vectors)
		#cluster
		# generate direction for cluster


	def plot(self):
		x, y = zip(*self.node_positions)
		plt.scatter(x,y)
		plt.show()

	def save_plot(self, name):
		x, y = zip(*self.node_positions)
		plt.scatter(x,y)
		plt.savefig(name)
		plt.clf()
