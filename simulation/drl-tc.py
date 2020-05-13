import numpy as np
import random

n_nodes = 5 

n_iterations = 100
n_episodes = 10
n_searches = 100
exploration_level = 0.5

n_actions = n_nodes**2 #action = connecting a node to the tree
n_states = n_nodes**(n_nodes-2) #state = unique spanning tree

states = np.zeros(n_states, n_nodes, n_nodes) #possible adjacency matrices
terminal_state_idxs  = []
visit_counts = np.zeros(n_states) # M(s)
visit_action_counts = np.zeros((n_states, n_actions)) # M(s,a)
state_action_values = np.zeros((n_states, n_actions)) # Q(s,a)
policy = np.zeros((n_states, n_actions)) # pi(s,a)


def dnn(self, state):
	pass

def mcts(state):
	# exit conditions of recursion
	if state.is_terminal():
		return r

	# expand to new search leaf
	if visit_counts[state_idx] == 0:
		state_policy, state_value = dnn(states[state_idx]) #policy --> probability for action (n_actions), value --> value of state = expected reward taking any trajectory from state
		invalid_actions = get_invalid_actions(state_idx) #valid actions --> indices of reachable states from current state
		state_policy[invalid_actions] = 0 #set invalid actions to zero probability
		state_policy /= state_policy.sum() #normalize over valid actions
		policy[state_idx] = state_policy
		visit_counts[state_idx] += 1
		return state_value

	# calculate UCBs
	upper_confidence_bounds = np.zeros(n_states)
	#for action in get_valid_actions(state_idx):
	#	upper_confidence_bounds[action] = state_action_values[state_idx, action] + exploration_level * policy[state_idx, action] * (np.)
	valid_action_idxs = get_valid_actions(state_idx)
	upper_confidence_bounds[valid_action_idxs] = state_action_values[state_idx, valid_action_idxs] + exploration_level * policy[state_idx, valid_action_idxs] * (np.sqrt(visit_counts[state_idx])/(1+visit_action_counts[state_idx, valid_action_idxs]))

	# choose action and recurse into transitioned state
	next_action_idx = np.random.choice(np.flatnonzero(upper_confidence_bounds == upper_confidence_bounds.max())) # not upper_confidence_bounds.argmax since it does not support random tie-breaking
	next_state_idx = next_action_idx # action=transition=state since we only have one action per state
	expected_value = mcts(next_state_idx) # since next state is selected using UCB the value of the leaf state is (comparable to?) the expected value


	# update tree statistics

	state_action_values[state_idx, next_action_idx] = (visit_action_counts[state_idx, next_action_idx] * state_action_values[state_idx, next_action_idx] + expected_value)/(visit_action_counts[state_idx, next_action_idx] + 1)
	visit_action_counts[state_idx, next_action_idx] += 1
	visit_counts[state_idx] += 1

	return expected_value

def get_valid_actions(state_idx):
	# take current state
	# filter actions that do not already have a 1 in adjacency triangle, an action could be represented as one-hot adjacency triangle and added as transition

def drltc():
	training_dataset = []
	for iteration in range(n_iterations):
		episode_root_state_idx = 0
		new_datasets = []
		for episode in range(n_episodes):
			visit_counts.fill(0)
			visit_action_counts.fill(0)
			state_action_values.fill(0.0)
			policy.fill(0.0)
			for search in range(n_searches):
				mcts(episode_root_state_idx)
			normalized_visits = visit_action_counts[episode_root_state_idx]/visit_action_counts[episode_root_state_idx].sum()
			new_datasets.append(states[episode_root_state_idx], normalized_visits, 0)
			if episode_root_state_idx in terminal_state_idxs:
				reward = simulate(states[episode_root_state_idx])
				for dataset in new_datasets:
					dataset[-1] = reward
			else:
				episode_root_state_idx = np.random.choice(n_nodes, p=normalized_visits) # sample an index according to action probabilities
		training_dataset += new_datasets
		random.shuffle(training_dataset)
		dnn_train()

