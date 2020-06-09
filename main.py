from MCTS import MCTS, State
from Model import DNN, Dataset
from Simulation import Simulation

import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch


n_nodes = 5
n_iterations = 100
n_episodes = 1
n_searches = 100
n_simulations = 50
n_trainings = 1
exploration_level = 5000
experiment = 'standard_variance'

def star_baseline(simulation):
	G = nx.generators.classic.star_graph(n_nodes-1)
	T = nx.bfs_tree(G, 0)
	topology = nx.convert_matrix.to_numpy_array(T)
	return simulation.eval(topology), T

def mst_baseline(simulation):
	# FIXME: MST baseline has worse lifetime than star baseline, maybe error in original paper?
	G = nx.Graph()
	for index, value in np.ndenumerate(simulation.node_distances):
		G.add_edge(*index, weight=value)
	T = nx.minimum_spanning_tree(G)
	T = nx.bfs_tree(T, 0)
	topology = nx.convert_matrix.to_numpy_array(T)
	return simulation.eval(topology), T

def random_baseline(simulation, n_simulations):
	lifetimes = []
	for i in range(n_simulations):
		G = nx.generators.trees.random_tree(n_nodes)
		T = nx.bfs_tree(G, 0)
		topology = nx.convert_matrix.to_numpy_array(T)
		lifetimes.append(simulation.eval(topology))
	lifetimes = np.array(lifetimes)
	return [lifetimes.mean(), lifetimes.max(), lifetimes.min(), lifetimes.std()]

def drltc(simulation):
	training_dataset = Dataset()
	dnn = DNN(n_nodes, minibatch=16, learning_rate=1e-6)
	statistics = []
	random_statistics = []
	max_trajectory = []

	for iteration in range(n_iterations):

		for episode in range(n_episodes):
			
			root_state = State(np.zeros((n_nodes, n_nodes)))

			for n in range(n_nodes):
				
				if root_state.is_terminal():
					reward = simulation.eval(root_state.adjacency)
					#print('reward', reward)
					for dataset in training_dataset.data[-n:]:
						dataset[-1] = np.array(reward) #update value in all datasets produced in this iteration
				else:	

					mcts = MCTS(root_state.shape, dnn, simulation, exploration_level)
					#TODO keep subtrees?

					for search in range(n_searches):
						print(f'\riteration {iteration:02}, episode {episode:02}, level {n:02}, search {search:02}', end='')
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
				max_topology = final_topology
		lifetimes = np.array(lifetimes)
		statistics.append([lifetimes.mean(), lifetimes.max(), lifetimes.min(), lifetimes.std()])
		random_statistics.append(random_baseline(simulation, n_simulations))
		print(f'statistics: {statistics[-1]}')
		#print(max_topology)

		if iteration%10 == 0 and iteration != 0:
			star, _ = star_baseline(simulation)
			mst, _ = mst_baseline(simulation)
			statistics_np = np.array(statistics, dtype=float)
			random_statistics_np = np.array(random_statistics, dtype=float)
			x = np.arange(statistics_np.shape[0])
			plt.plot(x, statistics_np[:,0], label='DRL-TC')
			plt.fill_between(x, statistics_np[:,0]-statistics_np[:,-2], statistics_np[:,0]+statistics_np[:,-2], alpha=0.5)
			plt.hlines(star, xmin=0, xmax=iteration, linestyles='dashed', label='Star')
			plt.hlines(mst, xmin=0, xmax=iteration, linestyles='dashdot', label='MST')
			plt.plot(x, random_statistics_np[:, 0], label='Random')
			plt.fill_between(x, random_statistics_np[:,0]-random_statistics_np[:,-2], random_statistics_np[:,0]+random_statistics_np[:,-2], alpha=0.5, linestyles='dotted')
			plt.ylabel('lifetime')
			plt.xlabel('iteration')
			plt.legend()
			plt.savefig(f'{experiment}/ckp_{experiment}_n{n_nodes}_e{n_episodes}_s{n_searches}_sim{n_simulations}_t{n_trainings}_i{iteration}.png')
			plt.clf()
			torch.save(dnn.model.state_dict, f'{experiment}/ckp_{experiment}_n{n_nodes}_e{n_episodes}_s{n_searches}_sim{n_simulations}_t{n_trainings}_i{iteration}.pt')

def check_dir():
	if os.path.isdir(f'./{experiment}'):
		raise Exception('Experiment exists')
	else:
		os.mkdir(f'./{experiment}')

if __name__ == '__main__':
	check_dir()
	simulation = Simulation(n_nodes)
	#simulation.plot()
	#_,star = star_baseline(simulation)
	#_,mst = mst_baseline(simulation)
	#random = random_baseline(simulation, n_simulations)
	#print(star, mst, random)
	#nx.draw(star, pos=simulation.node_positions)
	#plt.show()
	#nx.draw(mst, pos=simulation.node_positions)
	#plt.show()
	drltc(simulation)