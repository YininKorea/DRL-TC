import networkx as nx
import matplotlib.pyplot as plt

G = nx.generators.classic.star_graph(18)

nx.draw(G, with_labels=True, font_weight='bold')
plt.show()