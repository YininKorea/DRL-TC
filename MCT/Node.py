"""
This file is used to implement MCTS

* Terminal state: all nodes are connected to the tree

"""
p=0.5  ##a constant of power amplification in the link budget
E_p=0.1  ## energy dissipation per bit for data processing
import numpy as np
import torch

class State(object):
    def __init__(self,state,policy, value):
        self.state=state
        self.policy_matrix=policy
        self.value_matrix=value
    def game_end(self):
        for node in self.state:
            if torch.sum(node)<2:
                return False
        return True


class Node(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._childnode = {}  # a map from action to TreeNode
        self._visit_counts = 0
        self._data=0## its data
        self._Q = 0  ## the value of node
        self._u = 0 ## be used to compute UCB
        self._P = prior_p

    def select(self, c_puct):
        return max(self._childnode.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct): ## be used to compute the
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def expand(self, action_priors):  ## povide a possible action list (adding child node)
        """Node expansion.
        action_priors -- output from policy function - a list of tuples of actions
            and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = Node(self, prob)


    def update(self, leaf_value):  ## backpropogation
        """Update node values from leaf evaluation.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """resursively call update function of nodes.
        """
        # If it is not root, this node's parent should be updated first.

        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Check if it is a leaf node
        """
        return self._childnode == {}

    def is_root(self): ## root node checking
        return self._parent is None