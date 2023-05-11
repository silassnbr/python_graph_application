import random
import networkx as nx
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
G = nx.Graph()

with open('aa.txt', 'r') as f:
    dosya_icerigi = f.read()

cumleler = dosya_icerigi.split(".")

for i in range(len(cumleler)):
    G.add_node(cumleler[i],label=cumleler[i])

for i in range(len(cumleler) - 1):
    G.add_edge(cumleler[i],cumleler[i+1])


nx.draw(G, with_labels=True)
plt.show()
