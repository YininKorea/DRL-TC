from .Node import  Node,shape_transfer

import numpy as np
import copy
from Models import DL_model
import torch
p=50 ##a constant of power amplification in the link budget(for all sensors)
E_p=1 ## energy dissipation per bit for data processing

value_node=torch.from_numpy(np.random.randint(500,1000,(1,1,20,20))).float()
#value_node=torch.tensor(value_node, dtype=torch.float, device='cpu')## initialization of values for nodes
policy_matrix=torch.from_numpy(np.random.rand(1,1,20,20)).float()
#policy_matrix=torch.tensor(policy_matrix, dtype=torch.float, device='cpu')
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

## return a sequence of action probability and score with given state

'''
policy: 
1) check if there are any nodes not joined connection 
    if:
        then every unconnected nodes would be assigned a probability p_n
        for these have been connected nodes, they also have a probability p_u
        make sure P_n >P_u
'''





class MCTS(object):
    """implementation of Monte Carlo Tree Search.
        ## start state:
                 nineteen sensors randomly scattered in a circular area with a radius of 1000 m
                 Initial energy: 1j
                 Data transmission: uniformly generate [500,1000] bit each round.


    """

    def __init__(self,  n_action=19,c_puct=5, n_playout=5):
        """Arguments:
        policy_value_fn -- a function that takes in a board state and outputs a list of (action, probability)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from
            the current player's perspective) for the current player.
        c_puct -- a number of possible child node exploration, we set 19
        n_playout--- number of MCTS
        """
        self._root = Node(None, 1.0,0)
        self._policy = DL_model(in_channels=1).to('cpu')
        self._c_explation = n_action
        self.c_puct=c_puct  ## the ratio between exploration and refund, used to compute UCB
        self._n_playout = n_playout ## number of search
        self.state_record=[]  ## save the state

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        node = self._root
       # i=1
        while True:
            if node.is_leaf():
                break
            #print(node.index)
            current_index=node.index
            action, node = node.select(self._c_explation)
            state.do_move(current_index,action)

        current_state=shape_transfer(state.state)
        action_probs, leaf_value = self._policy(current_state)## get predictions from DNN
        action_probs,leaf_value=shape_transfer(action_probs),shape_transfer(leaf_value)  ## squeeze outputs from DNN into valid actions and values
        #print(leaf_value.shape)
        end = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            self.state_record.append(state.adjacency_matrix)
        leaf_value=-leaf_value[node.index][node.index]
        #print(leaf_value.shape)
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the available actions and their corresponding probabilities
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities
        """
        for n in range(self._n_playout):
            #print(n)
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on the visit counts at the root node
        act_visits = [(act, node._visit_counts) for act, node in self._root._childnode.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(visits))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = Node(None, 1.0)

    def __str__(self):
        return self.state_record