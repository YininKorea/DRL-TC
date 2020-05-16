import numpy as np
from collections import defaultdict as ddict

class MCTS:

	def __init__(self, dnn, exploration_level):
		self.Q = dict()
		self.pi = dict()
		self.visits = dict()
		self.action_visits = dict()
		self.dnn = dnn
		self.exploration_level = exploration_level

	def search(self, state):

		if state.is_terminal():
			state_policy, state_value = self.dnn.eval(state.adjacency)
			return state_value # could also return reward == simulation

		if state not in self.visits:
			state_policy, state_value = self.dnn.eval(state.adjacency)
			state_policy[~state.get_valid_actions()] = 0 # set invalid actions to 0
			state_policy /= state_policy.sum() # re-normalize over valid actions
			self.pi[state] = state_policy
			self.visits[state] = 1
			self.action_visits[state] = np.zeros(state.adjacency.shape)

			return state_value

		valid_actions = state.get_valid_actions()
		upper_confidence_bounds = np.zeros(state.adjacency.shape)
		upper_confidence_bounds[valid_actions] = self.Q[state] + self.exploration_level * self.pi[state] * (np.sqrt(self.visits[state])/(1+self.action_visits[state]))

		next_action = tiebreak_argmax(upper_confidence_bounds) # not normal argmax since it does not support random tie-breaking
		next_state = state.transition(next_action)

		expected_value = self.search(next_state)

		self.Q[state][next_action] = (self.action_visits[state][next_action] * self.Q[state][next_action] + expected_value) / (self.action_visits[state][next_action] + 1)
		self.action_visits[state][next_action] += 1
		self.visits[state] += 1

		return expected_value


class State:

	def __init__(self, n_nodes, adjacency=None):
		self.adjacency = adjacency or np.zeros((n_nodes, n_nodes))
		self.n_nodes = n_nodes

	def is_terminal(self):
		return self.adjacency.sum() == self.n_nodes-1

	def get_valid_actions(self):
		not_connected = np.all(self.adjacency == 0, axis=0)
		not_connected[0] = False # gateway root node never has a parent
		return np.outer(~not_connected, not_connected) # bool mask for adjacency matrix

	def transition(self, action):
		adjacency = self.adjacency.copy()
		adjacency[action] = 1
		return State(self.n_nodes, adjacency)

	def __hash__(self):
		return hash(str(self.adjacency))


def tiebreak_argmax(array):
	#returns argmax in multi-dimensional index with random tie-breaking 
	return np.unravel_index(np.random.choice(np.flatnonzero(array == array.max())), shape=array.shape)