import numpy as np

class State:

	def __init__(self, n_nodes):
		self.n_nodes = n_nodes
		self.adjacency = np.zeros((n_nodes, n_nodes))

	def is_terminal(self):
		return self.adjacency.sum() == self.n_nodes-1