#####################################################################
# In this example we will explore multiple statistics for our
# graph
#
# We will also transform our graph into a PyTorch tensor,
# so that we can make some predictions based on our graph
#
# Then we will discuss using graphs for learning,
# with a node embedding model
#
# For more complex examples check out:
#  - DeepWalk
#  - node2vec
#  - Casual Anonymous Walks (CAW)

#####################################################################
# Getting started
#
# We will use a popular example (network) graph
# The Karate Club Network, but without Ralph Macchio

import argparse
import matplotlib.pyplot as plt
import networkx as nx

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-s",
                    "--save",
                    help="Save graphs instead of showing them",
                    action="store_true")

args = parser.parse_args()
######################################################################

social_graph = nx.karate_club_graph()

# Remember the types of graphs from our last example?
print(type(social_graph))

# Since this graph represents the social connections from a
# college karate club, it makes sense that our graph is
# undirected!

# Lets visualize the relationships in this organization
nx.draw_circular(social_graph, cmap=plt.cm.Greens, with_labels=True)

if args.save:
    plt.savefig("Karate_Club_Graph.png")
    plt.clf()
else:
    plt.show()

#####################################################################
# Lets begin making observations about this graph

# We can find the average degree of our nodes
# By taking the number of edges divded by the number of nodes
def avg_node_degree(num_edges, num_nodes):
    return round(num_edges / num_nodes)

# Remember how to get the number of edges / nodes in our graph?
num_edges = social_graph.number_of_edges()
num_nodes = social_graph.number_of_nodes()
print("# of Nodes: {0}\n# of Edges: {1}\nAverage Node Degree: {2}"
      .format(num_nodes, num_edges, avg_node_degree(num_edges, num_nodes)))

# What might this tell us?
# Well for our example graph it might show on average the number
# of other members of the club that each person interacts with
# outside of the organization
# Lets remember how important it can be to know about the data
# we are working with!

# What do you think finding the
# average clustering coefficient might tell us?
def avg_cluster_coef(Graph):

    # False so we do not include non zero clustered nodes
    return(nx.average_clustering(Graph, count_zeros=False))

# Think about what the clustering of the graph will tell us
# If we consider "cliques" this may show us a relation
# Cliques, in terms of our graph can relate to the subset
# of nodes which are connected
print("\nAverage Cluster Coefficient: ")
print(avg_cluster_coef(social_graph))
print("\n")

# Now consider PageRank,
# Think of this as the popularity of someone in the group
#
# In web apges this might tell us which page sees the most
# traffic
print("Node Page Ranks: ")
popularity = nx.pagerank(social_graph, alpha=0.8)
print(popularity)
print("\n")
# Finally, the Closeness Centrality of a node is a measure
# of centrality in a network
# We find this as the sum of the length of the shortest
# paths between the node and all other nodes in our graph
#
# The more central a node is, the closer it is to all
# other nodes in the graph
print("Closeness Centralities: ")
print(nx.closeness_centrality(social_graph))
print("\n")
