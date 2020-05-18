from Simulation import Simulation
import networkx as nx
#import matplotlib.pyplot as plt

n_nodes = 5

G = nx.generators.classic.star_graph(n_nodes-1)
simulation = Simulation(n_nodes)
topology = nx.convert_matrix.to_numpy_array(G)
topology[:,0] = 0
print(simulation.eval(topology))
#nx.draw(G, with_labels=True, font_weight='bold')
#plt.show()