from MCTS import MCTS, State
from Model import DNN, Dataset
from Simulation import Simulation

import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import random
import torch


n_nodes = 5
n_iterations = 100
n_episodes = 5
n_searches = 100
n_simulations = 20
exploration_level = n_nodes#*1000

def star_baseline(simulation):
	G = nx.generators.classic.star_graph(n_nodes-1)
	topology = nx.convert_matrix.to_numpy_array(G)
	topology[:,0] = 0
	return simulation.eval(topology)

def drltc(simulation):
	training_dataset = Dataset()
	dnn = DNN(n_nodes, minibatch=16, learning_rate=10e-6)
	statistics = []

	for iteration in range(n_iterations):
		episode_root_state = State(np.zeros((n_nodes, n_nodes)))

		for episode in range(n_episodes):
			mcts = MCTS(episode_root_state.shape, dnn, simulation, exploration_level)

			for search in range(n_searches):
				print(f'\riteration {iteration}, episode {episode}, search {search}', end='')
				reward = mcts.search(episode_root_state)

			if episode_root_state.is_terminal():
				state_value = reward
			else:
				q = mcts.Q[episode_root_state]
				state_value = q.sum()/(q != 0).sum()

			# update
			print('\nexploration:', np.linalg.norm(mcts.action_visits[episode_root_state].flatten(), 0))
			if mcts.action_visits[episode_root_state].sum() != 0:
				normalized_visits = mcts.action_visits[episode_root_state]/mcts.action_visits[episode_root_state].sum()
			else:
				normalized_visits = mcts.action_visits[episode_root_state]
			training_dataset.add([episode_root_state.adjacency, normalized_visits.flatten(), np.array(state_value)])
			print(state_value)
			if episode_root_state.is_terminal():
				reward = simulation.eval(episode_root_state.adjacency)/1000
				#print('reward', reward)
				for dataset in training_dataset.data[-episode:]:
					dataset[-1] = np.array(reward) #update value in all datasets produced in this iteration
			else:
				#print(normalized_visits)
				next_action = np.unravel_index(np.random.choice(n_nodes**2, p=normalized_visits.flatten()), shape=normalized_visits.shape)
				episode_root_state = episode_root_state.transition(next_action)
		print('\n')
		dnn.train(training_dataset)	

		# construct test topologies
		lifetimes = []
		for i in range(n_simulations):
			state = State(np.zeros((n_nodes, n_nodes)))
			while not state.is_terminal():
				state_policy, _ = dnn.eval(state.adjacency)
				state_policy[~state.get_valid_actions()] = 0 # set invalid actions to 0
				state_policy /= state_policy.sum() # re-normalize over valid actions
				next_action = np.unravel_index(np.random.choice(n_nodes**2, p=state_policy.flatten()), shape=state_policy.shape)
				state = state.transition(next_action)
			final_topology = state.adjacency
			lifetimes.append(simulation.eval(final_topology))
		statistics.append([sum(lifetimes)/n_simulations, max(lifetimes), min(lifetimes), max(lifetimes)-min(lifetimes)])
		print(f'statistics: {statistics[-1]}')
		#torch.save(lifetimes, f'lifetimes_iteration{iteration}.pt')

		if iteration%30 == 0 and iteration != 0:
			statistics_np = np.array(statistics)
			plt.plot(statistics_np[:,0])
			plt.plot(statistics_np[:,1])
			plt.plot(statistics_np[:,2])
			plt.ylabel('lifetime')
			plt.xlabel('iteration')
			plt.show()
	#torch.save(dnn.model.state_dict, 'model_checkpoint_2.pt')

if __name__ == '__main__':
	simulation = Simulation(n_nodes)
	print(star_baseline(simulation))
	drltc(simulation)