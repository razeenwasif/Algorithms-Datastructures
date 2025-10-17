import osmnx as ox 
import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import time

ox.settings.use_cache = True
# Define the location using a place name or coordinates
place_name = "Palo Alto, California, USA"
graph = ox.graph_from_place(place_name, network_type="all")

# Convert the graph to a networkx MultiDiGraph for further analysis
graph = nx.MultiDiGraph(graph)

# Specify the origin and destination points
orig = ox.distance.nearest_nodes(graph, -122.1665, 37.4292)  # Example coordinates
dest = ox.distance.nearest_nodes(graph, -122.1819, 37.4185)  # Example coordinates

# Calculate the shortest path
route = nx.shortest_path(graph, orig, dest, weight="length")

print("Shortest Path:", route)

fig, ax = plt.subplots(figsize=(30, 14))
fig.patch.set_facecolor('black')
ox.plot_graph(ox.project_graph(graph), ax=ax, edge_color='lavender',
                        node_size=5, show=False, close=False, node_color='pink')
ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.show()
