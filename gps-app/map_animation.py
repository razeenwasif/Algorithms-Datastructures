import osmnx as ox 
import networkx as nx 
from manim import *

class MapAnimation(Scene):
    def construct(self):
        # Step 1: Get the street network 
        placeName = "Manhattan, New York, USA"
        G = ox.graph_from_place(placeName, network_type='drive')

        # Step 2: Convert to nodes and edges for Manim 
        pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
        
        # Normalize and scale the position to fit into the scene
        x_coords = [x for x, y in pos.values()]
        y_coords = [y for x, y in pos.values()]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords) 

        # Apply normalization to fit into a reasonable range for Manim
        pos = {node: [(data['x'] - x_min) / (x_max - x_min) * 10, 
                      (data['y'] - y_min) / (y_max - y_min) * 10] 
               for node, data in G.nodes(data=True)}
        

        edges = [(u, v) for u, v in G.edges()]

        # Step 3: Create Manim objects for edges 
        edge_lines = []
        for u, v in edges:
            line = Line(start=[pos[u][0], pos[u][1], 0], end=[pos[v][0], pos[v][1], 0])
            edge_lines.append(line)

        # Step 4: Create a visual representation
        map_group = VGroup(*edge_lines)

        # Step 5: Animate the visualization
        self.play(Create(map_group))
        self.wait(2)

        # Optionally highlight a specific edge 
        if edges:
            highlight_edge = Line(start=[pos[edges[0][0]][0], pos[edges[0][0]][1], 0], 
                      end=[pos[edges[0][1]][0], pos[edges[0][1]][1], 0], 
                      color=RED)
            
            self.play(Create(highlight_edge))
            self.wait(2)
        self.play(FadeOut(map_group), FadeOut(highlight_edge))

    # To run the script, use the command:
    # manim -pql map_animation.py MapAnimation
