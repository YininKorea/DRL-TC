import numpy as np
from MCT import  MCTS
from MCT import Node,State
import torch


value_node=torch.from_numpy(np.random.randint(500,1000,(1,1,20,20))).float()
policy_matrix=torch.from_numpy(np.random.rand(1,1,20,20)).float()

'''
policy: 
1) check if there are any nodes not joined connection 
    if:
        then every unconnected nodes would be assigned a probability p_n
        for these have been connected nodes, they also have a probability p_u
        make sure P_n >P_u
'''

# def original_policy_builter(state):
#     action_probability=[]
#     leaf_value=[]
#     for indice, node in enumerate(state[1:]):
#         if sum(node) < 2:
#             action_probability.append(1) ##collect nodes that have not been connected:
#             leaf_value.append(value_node[indice])
#         else:
#             action_probability.append(0)
#             leaf_value.append(value_node[indice])
#
#     return action_probability,leaf_value

def test():
    #print(value_node)
    #print(policy_matrix)
    initial_state=State(torch.from_numpy(np.identity(20)),policy_matrix,value_node) ## an identification matrix (without any connection, the first row stands for the root node)
    MCTS_tree=MCTS()

    MCTS_tree.get_move_probs(initial_state)






if __name__=='__main__':
    test()