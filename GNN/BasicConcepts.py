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

#####################################################################
# Guide made with help from Stanford's course CS224W
# Thanks!
#####################################################################

#####################################################################
# Networkx
#
# This will help us with our different graph types
#
# Graph types (from Networkx)
#
# # Class Name #    Type    # Self-Loops # Parallel Edges #
###########################################################
#    Graph     # undirected #    Yes     #      No        #
###########################################################
#   DiGraph    #  directed  #    Yes     #      No        #
###########################################################
#  MultiGraph  # undirected #    Yes     #      Yes       #
###########################################################
# MultiDiGraph #  directed  #    Yes     #      Yes       #
###########################################################

# Create an undirected graph
no_direction = nx.Graph()
print(no_direction.is_directed())

# A directed graph (please enjoy our clever naming)
one_direction = nx.DiGraph()
print(one_direction.is_directed())

# Here we add a graph level attribute
# to our unidrected graph
no_direction.graph["Name"] = "Bar"
print(no_direction.graph)

#####################################################################
# Nodes
#
# Adding nodes to our graph is simple

# Here we add a node level attribute
no_direction.add_node(0, feature=0, label=0)

# Getting attributes of the Node 0
node_0_attr = no_direction.nodes[0]
print("Node 0 attributes: {}".format(node_0_attr))

# Adding multiple nodes with data
# nodes will have format of: (# { "key": value, "key": value })

no_direction.add_nodes_from([
    (1, {"feature": 1, "label": 1}),
    (2, {"feature": 2, "label": 2}),
])

# We can loop through all nodes in the graph easily
# (set data=True) to print attributes
for node in no_direction.nodes(data=True):
    print(node)


# Get number of nodes in the graph
print("Our graph has {} nodes".format(no_direction.number_of_nodes()))

#####################################################################
# Edges
#
# Adding edges (connections between nodes) can be done as so

# Add an edge between nodes 0 - 1, with a weight
no_direction.add_edge(0, 1, weight=0.5)

# Get attributes of the edge between nodes 0 - 1
edge_0_to_1 = no_direction.edges[(0, 1)]
print("The edge (0, 1) has the attributes: {}".format(edge_0_to_1))

# We can add multiple edges just as we did nodes
no_direction.add_edges_from([
    (1, 2, {"weight": 0.3}),
    (2, 0, {"weight": 0.1}),
])

# Edges can also be looped through like nodes
#(But we do not need to use data=True)
for edge in no_direction.edges():
    print(edge)

# Get number of edges in the graph
print("Our graph has {} edges".format(no_direction.number_of_edges()))

#####################################################################
# Visualization
#
# Let us see our graph

# nx.draw uses matplotlib so remember to call plt.show()
nx.draw(no_direction, with_labels=True)

if args.save:
    plt.savefig("Basic_Graph1.png")
    plt.clf()
else:
    plt.show()

#####################################################################
# Node Degree and Neighbor
#
node_id = 1

# Degree of node 1
print("Node {0} has a degree of {1}".format(node_id,
                                            no_direction.degree[node_id]
))

# Who are node 1's neighbors?
for neighbor in no_direction.neighbors(node_id):
    print("Node {0} has neighbor(s) {1}".format(node_id, neighbor))

#####################################################################
# Additional Functions
#

# Making a new path-like graph to morph into a directed graph
# (Pass the number of nodes to create)
top = nx.DiGraph(nx.path_graph(5))
nx.draw(top, with_labels=True)

if args.save:
    plt.savefig("Directed_Graph1.png")
    plt.clf()
else:
    plt.show()

# Pagerank
top_rank = nx.pagerank(top, alpha=0.8)
print("Node popularity: {}".format(top_rank))

#####################################################################
# We can create very interesting graph layouts and styles

G = nx.star_graph(20)
pos = nx.spring_layout(G)
colors = range(20)
options = {
    "node_color": "#C32F27",
    "edge_color": colors,
    "width": 4,
    "edge_cmap": plt.cm.Wistia,
    "with_labels": True,
}
nx.draw(G, pos, **options)

if args.save:
    plt.savefig("Star_Graph.png")
    plt.clf()
else:
    plt.show()


