from MCTS import MCTS, State
from Model import DNN, Dataset
from Simulation import Simulation

import matplotlib.pyplot as plt
import numpy as np
import random
import torch

n_nodes = 10
n_iterations = 20
n_episodes = 10
n_searches = 50
exploration_level = 0.5

def drltc():
	training_dataset = Dataset()
	dnn = DNN(n_nodes, minibatch=16, learning_rate=10e-6)
	simulation = Simulation(n_nodes)
	statistics = []

	for iteration in range(n_iterations):
		episode_root_state = State(np.zeros((n_nodes, n_nodes)))

		for episode in range(n_episodes):
			mcts = MCTS(episode_root_state.shape, dnn, simulation, exploration_level)

			for search in range(n_searches):
				print(f'\riteration {iteration}, episode {episode}, search {search}', end='')
				state_value = mcts.search(episode_root_state)

			if mcts.action_visits[episode_root_state].sum() != 0:
				normalized_visits = mcts.action_visits[episode_root_state]/mcts.action_visits[episode_root_state].sum()
			else:
				normalized_visits = mcts.action_visits[episode_root_state]
			training_dataset.add([episode_root_state.adjacency, normalized_visits.flatten(), np.array(state_value)])

			if episode_root_state.is_terminal():
				reward = simulation.eval(episode_root_state.adjacency)
				for dataset in training_dataset.data[-episode:]:
					dataset[-1] = np.array(reward) #update value in all datasets produced in this iteration
			else:
				next_action = np.unravel_index(np.random.choice(n_nodes**2, p=normalized_visits.flatten()), shape=normalized_visits.shape)
				episode_root_state = episode_root_state.transition(next_action)
		print('\n')
		dnn.train(training_dataset)

		# construct test topology
		lifetimes = []
		for i in range(10):
			state = State(np.zeros((n_nodes, n_nodes)))
			while not state.is_terminal():
				state_policy, _ = dnn.eval(state.adjacency)
				state_policy[~state.get_valid_actions()] = 0 # set invalid actions to 0
				state_policy /= state_policy.sum() # re-normalize over valid actions
				next_action = np.unravel_index(np.random.choice(n_nodes**2, p=state_policy.flatten()), shape=state_policy.shape)
				state = state.transition(next_action)

			final_topology = state.adjacency
			lifetimes.append(simulation.eval(final_topology))
		statistics.append([sum(lifetimes)/10, max(lifetimes), min(lifetimes)])
		print(f'statistics: {statistics[-1]}')

	statistics = np.array(statistics)
	plt.plot(statistics[:,0])
	plt.plot(statistics[:,1])
	plt.plot(statistics[:,2])
	plt.ylabel('lifetime')
	plt.xlabel('iteration')
	plt.show()
	torch.save(dnn.model.state_dict, 'model_checkpoint_2.pt')

if __name__ == '__main__':
	drltc()