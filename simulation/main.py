from MCTS import MCTS, State
from Model import DNN, Dataset
from Simulation import Simulation

import numpy as np
import random

n_nodes = 5
n_iterations = 10
n_episodes = 4
n_searches = 10
exploration_level = 0.5

def drltc():
	training_dataset = Dataset()
	dnn = DNN(n_nodes, minibatch=16, learning_rate=10e-6)
	simulation = Simulation(n_nodes)

	for iteration in range(n_iterations):
		episode_root_state = State(np.zeros((n_nodes, n_nodes)))

		for episode in range(n_episodes):
			mcts = MCTS(episode_root_state.shape, dnn, simulation, exploration_level)

			for search in range(n_searches):
				print(f'\repisode {episode}, search {search}', end='')
				state_value = mcts.search(episode_root_state)

			if mcts.action_visits[episode_root_state].sum() != 0:
				normalized_visits = mcts.action_visits[episode_root_state]/mcts.action_visits[episode_root_state].sum()
			else:
				normalized_visits = mcts.action_visits[episode_root_state]
			training_dataset.add((episode_root_state.adjacency, normalized_visits.flatten(), np.array(state_value)))

			if episode_root_state.is_terminal():
				reward = simulation.eval(episode_root_state.adjacency)
				for dataset in training_dataset.data[-episode:]:
					dataset[-1] = np.array(reward) #update value in all datasets produced in this iteration
			else:
				next_action = np.unravel_index(np.random.choice(n_nodes**2, p=normalized_visits.flatten()), shape=normalized_visits.shape)
				episode_root_state = episode_root_state.transition(next_action)
		dnn.train(training_dataset)

		# construct test topology
		# state = State(n_nodes)
		# while not state.is_terminal():
		# 	state_policy, _ = dnn.eval(state.adjacency)
		# 	state_policy[~state.get_valid_actions()] = 0 # set invalid actions to 0
		# 	state_policy /= state_policy.sum() # re-normalize over valid actions
		# 	next_action = np.unravel_index(np.random.choice(n_nodes**2, p=state_policy.flatten()), shape=state_policy.shape)
		# 	state = state.transition(next_action)

		# final_topology = state.adjacency

if __name__ == '__main__':
	drltc()