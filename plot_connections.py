import dash
import dash_html_components as html
# import dash_core_components as dcc
from dash import dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import networkx as nx
import pylab as plt

import numpy as np
import pdb

from analysis import cov_space_and_time

num_nodes = 512*2
#create graph G
G = nx.Graph()

x = np.linspace(0, 7, 8)
xyz = np.meshgrid(x, x, x)
block_pos = np.vstack(map(np.ravel, xyz)).T
block_pos = block_pos[:,[1,0,2]] * 16

# for i in range(512):
#     G.add_node(i+512, pos=(i,2))

_, sp_cov_sum = cov_space_and_time()
# print(t_cov_sum.shape, sp_cov_sum.shape, np.min(sp_cov_sum), np.max(sp_cov_sum))

# plt.hist(sp_cov_sum, density=True, bins=30)
# plt.show()

# G = nx.grid_graph(dim=[8, 8, 8])
# pdb.set_trace()
# filter out, if spatial connection is smaller than threshold=0.5
# for i in range(512):
#     G.add_node(i, pos=block_pos[i], ts=1)

cnt=0
edge_widths = []
target = [80,32,64]
index = block_pos.tolist().index(target)
print(index)

G.add_node(index, pos=block_pos[index])

for i in range(512):
    # for j in range(i, 512):
    #     if sp_cov_sum[i, j]>0.8:
    #         G.add_node(i, pos=block_pos[i])
    #         G.add_node(j, pos=block_pos[j])
    #         G.add_edge(i, j, weight=sp_cov_sum[i, j])
    #         edge_widths.append(sp_cov_sum[i, j])
    G.add_node(i, pos=block_pos[i])
    if sp_cov_sum[i, index]>0.5:
        G.add_edge(i, index, weight=sp_cov_sum[i, index])
        edge_widths.append(sp_cov_sum[i, index])

# to_remove=[ n for n,d in G.degree_iter(with_labels=True) if d==0 ]
# # to_remove = [n for n in outdeg if outdeg[n] < 1]
# G.remove_nodes_from(to_remove)
Num_nodes = len(G.nodes)
edges = G.edges()
# node_pos = nx.spring_layout(G, dim = 3, k = 0.5) # k regulates the distance between nodes
node_pos=nx.get_node_attributes(G,'pos')


x_nodes= [node_pos[key][0] for key in node_pos.keys()] # x-coordinates of nodes
y_nodes = [node_pos[key][1] for key in node_pos.keys()] # y-coordinates
z_nodes = [node_pos[key][2] for key in node_pos.keys()] # z-coordinates

#we need to create lists that contain the starting and ending coordinates of each edge.
x_edges=[]
y_edges=[]
z_edges=[]

#create lists holding midpoints that we will use to anchor text
xtp = []
ytp = []
ztp = []

# pdb.set_trace()
#need to fill these with all of the coordinates
for edge in edges:
    #format: [beginning,ending,None]
    x_coords = [node_pos[edge[0]][0],node_pos[edge[1]][0],None]
    x_edges += x_coords
    xtp.append(0.5*(node_pos[edge[0]][0]+ node_pos[edge[1]][0]))

    y_coords = [node_pos[edge[0]][1],node_pos[edge[1]][1],None]
    y_edges += y_coords
    ytp.append(0.5*(node_pos[edge[0]][1]+ node_pos[edge[1]][1]))

    z_coords = [node_pos[edge[0]][2],node_pos[edge[1]][2],None]
    z_edges += z_coords
    ztp.append(0.5*(node_pos[edge[0]][2]+ node_pos[edge[1]][2])) 


etext = [f'weight={w}' for w in edge_widths]

trace_weights = go.Scatter3d(x=xtp, y=ytp, z=ztp,
    mode='markers',
    marker =dict(color='rgb(125,125,125)', size=1), #set the same color as for the edge lines
    text = etext, hoverinfo='text')

#create a trace for the edges
trace_edges = go.Scatter3d(
    x=x_edges,
    y=y_edges,
    z=z_edges,
    mode='lines',
    line=dict(color='black', width=2),
    hoverinfo='all')

#create a trace for the nodes
trace_nodes = go.Scatter3d(
    x=x_nodes,
    y=y_nodes,
    z=z_nodes,
    mode='markers',
    hoverinfo='x+y+z',
    marker=dict(symbol='circle',
            size=10,
            color='skyblue')
    )

#Include the traces we want to plot and create a figure
data = [trace_edges, trace_nodes, trace_weights]
fig = go.Figure(data=data)

fig.show()

# pos = nx.spring_layout(G, pos=fixed_positions, fixed = fixed_nodes)
# nx.draw_networkx(G, pos)
# pos = nx.spring_layout(G, iterations=20)
# nx.draw(G, node_pos, with_labels=True, node_size=4, width=edge_widths)
# plt.show()