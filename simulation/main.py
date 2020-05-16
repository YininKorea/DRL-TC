from MCTS import MCTS, State
from Model import DNN

n_nodes = 20
n_iterations = 100
n_episodes = 10
n_searches = 100
exploration_level = 0.5

def drltc():
	training_dataset = []
	dnn = DNN(minibatch=16, learning_rate=10e-6)

	for iteration in range(n_iterations):
		episode_root_state = State(n_nodes)

		for episode in range(n_episodes):
			mcts = MCTS(dnn, exploration_level)

			for search in range(n_searches):
				mcts.search(episode_root_state)

			normalized_visits = mcts.action_visits[episode_root_state]/mcts.action_visits[episode_root_state].sum()
			training_dataset.append(episode_root_state.adjacency, normalized_visits.flatten(), 0)

			if episode_root_state.is_terminal():
				reward = simulate(episode_root_state.adjacency)
				for dataset in new_datasets:
					dataset[-1] = reward
			else:
				next_action = np.unravel_index(np.random.choice(n_nodes**2, p=normalized_visits.flatten()), shape=normalized_visits.shape)
				episode_root_state = episode_root_state.transition(next_action)
		training_dataset += new_datasets
		random.shuffle(training_dataset)
		dnn.train(training_dataset)

		# construct test topology
		state = State(n_nodes)
		while not state.is_terminal():
			state_policy, _ = dnn.eval(state.adjacency)
			state_policy[~state.get_valid_actions()] = 0 # set invalid actions to 0
			state_policy /= state_policy.sum() # re-normalize over valid actions
			next_action = np.unravel_index(np.random.choice(n_nodes**2, p=state_policy.flatten()), shape=state_policy.shape)
			state = state.transition(next_action)

		final_topology = state.adjacency

if __name__ == '__main__':
	drltc()