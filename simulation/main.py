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
n_episodes = 1
n_searches = 100
n_simulations = 20
n_trainings = 10
exploration_level = 5

def star_baseline(simulation):
	G = nx.generators.classic.star_graph(n_nodes-1)
	topology = nx.convert_matrix.to_numpy_array(G)
	topology[:,0] = 0
	return simulation.eval(topology)

def drltc(simulation):
	training_dataset = Dataset()
	dnn = DNN(n_nodes, minibatch=16, learning_rate=10e-6)
	statistics = []
	max_trajectory = []

	for iteration in range(n_iterations):
		root_state = State(np.zeros((n_nodes, n_nodes)))

		for episode in range(n_episodes):

			for n in range(n_nodes):

				
				if root_state.is_terminal():
					reward = simulation.eval(root_state.adjacency)/1000
					#print('reward', reward)
					for dataset in training_dataset.data[-n:]:
						dataset[-1] = np.array(reward) #update value in all datasets produced in this iteration
				else:	

					mcts = MCTS(root_state.shape, dnn, simulation, exploration_level)

					for search in range(n_searches):
						print(f'\riteration {iteration}, episode {episode}, level {n}, search {search}', end='')
						mcts.search(root_state)

					# update
					print('\nexploration:', np.linalg.norm(mcts.action_visits[root_state].flatten(), 0))
					if mcts.action_visits[root_state].sum() != 0:
						normalized_visits = mcts.action_visits[root_state]/mcts.action_visits[root_state].sum()
					else:
						normalized_visits = mcts.action_visits[root_state]

					training_dataset.add([root_state.adjacency, normalized_visits.flatten(), None])
					#print(normalized_visits)
					next_action = np.unravel_index(np.random.choice(n_nodes**2, p=normalized_visits.flatten()), shape=normalized_visits.shape)
					root_state = root_state.transition(next_action)

			#for s in max_trajectory:
				#print(s.adjacency)
				#print(mcts.action_visits[s])
		print('\n')
		for i in range(n_trainings):
			dnn.train(training_dataset)

		# construct test topologies
		lifetimes = []
		max_value = 0
		for i in range(n_simulations):
			state = State(np.zeros((n_nodes, n_nodes)))
			trajectory = [state]
			while not state.is_terminal():
				state_policy, _ = dnn.eval(state.adjacency)
				#print(state_policy)
				state_policy[~state.get_valid_actions()] = 0 # set invalid actions to 0
				state_policy /= state_policy.sum() # re-normalize over valid actions
				next_action = np.unravel_index(np.random.choice(n_nodes**2, p=state_policy.flatten()), shape=state_policy.shape)
				state = state.transition(next_action)
				trajectory.append(state)
			final_topology = state.adjacency
			value = simulation.eval(final_topology)
			lifetimes.append(value)
			if (value > max_value):
				max_value = value
				max_trajectory = trajectory
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
	#simulation.plot()
	print(star_baseline(simulation))
	drltc(simulation)