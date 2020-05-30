import numpy as np
from collections import defaultdict as ddict
from functools import partial

class MCTS:

	def __init__(self, state_shape, dnn, simulation, exploration_level):
		self.Q = ddict(partial(np.zeros, state_shape))
		self.pi = dict()
		self.visits = dict()
		self.action_visits = ddict(partial(np.zeros, state_shape))
		self.dnn = dnn
		self.simulation = simulation
		self.exploration_level = exploration_level

	def search(self, state):
		if state.is_terminal():
			#reward = self.simulation.eval(state.adjacency)/1000
			_, reward = self.dnn.eval(state.adjacency)
			return reward

		if state not in self.visits:
			state_policy, state_value = self.dnn.eval(state.adjacency)
			state_policy[~state.get_valid_actions()] = 0 # set invalid actions to 0
			state_policy /= state_policy.sum() # re-normalize over valid actions
			self.pi[state] = state_policy
			#print('eval', state_policy, state_value)
			self.visits[state] = 1

			return state_value

		valid_actions = state.get_valid_actions()
		upper_confidence_bounds = self.Q[state] + self.exploration_level * self.pi[state] * (np.sqrt(self.visits[state])/(1+self.action_visits[state]))
		upper_confidence_bounds[~valid_actions] = 0

		next_action = tiebreak_argmax(upper_confidence_bounds) # not normal argmax since it does not support random tie-breaking
		next_state = state.transition(next_action)

		expected_value = self.search(next_state)

		self.Q[state][next_action] = (self.action_visits[state][next_action] * self.Q[state][next_action] + expected_value) / (self.action_visits[state][next_action] + 1)
		self.action_visits[state][next_action] += 1
		self.visits[state] += 1

		return expected_value


class State:

	def __init__(self, adjacency):
		self.adjacency = adjacency
		self.shape = adjacency.shape
		self.n_nodes = adjacency.shape[0]

	def is_terminal(self):
		return self.adjacency.sum() == self.n_nodes-1

	def get_valid_actions(self):
		# adjacency matrix has parents in columns --> empty columns are unconnected nodes
		not_connected = np.all(self.adjacency == 0, axis=0)
		not_connected[0] = False # gateway root node is never unconnected, never valid action
		# valid means connecting unconnected node (zero column) to connected node (nonzero column)
		# except themselves --> logical AND between zero columns and its complement
		return np.outer(~not_connected, not_connected) # bool mask for adjacency/transition matrix

	def transition(self, action):
		adjacency = self.adjacency.copy()
		adjacency[action] = 1
		return State(adjacency)

	def __str__(self):
		return str(self.adjacency)

	def __hash__(self):
		return hash(str(self.adjacency))

	def __eq__(self, other):
		return np.all(self.adjacency == other.adjacency)

def tiebreak_argmax(array):
	#returns argmax in multi-dimensional index with random tie-breaking 
	return np.unravel_index(np.random.choice(np.flatnonzero(array == array.max())), shape=array.shape)