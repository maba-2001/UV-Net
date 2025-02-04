import argparse
import numpy as np
from occwl.viewer import Viewer
from occwl.io import load_step
from occwl.edge import Edge
from occwl.solid import Solid

import torch
import dgl
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
from matplotlib import cm


def draw_face_uvgrids(solid, graph, viewer):
    face_uvgrids = graph.ndata["x"].view(-1, 7)
    points = []
    normals = []
    importance_values = graph.ndata["importance"].cpu().numpy()  # Assuming importance is stored in 'importance'
    norm = plt.Normalize(vmin=np.min(importance_values), vmax=np.max(importance_values))  # Normalize importance

    for idx in range(face_uvgrids.shape[0]):
        # Don't draw points outside trimming loop
        if face_uvgrids[idx, -1] == 0:
            continue
        points.append(face_uvgrids[idx, :3].cpu().numpy())
        normals.append(face_uvgrids[idx, 3:6].cpu().numpy())

    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)

    bbox = solid.box()
    max_length = max(bbox.x_length(), bbox.y_length(), bbox.z_length())

    # Use a colormap to map the importance to colors
    colors = cm.viridis(norm(importance_values))[:, :3]  # Using viridis colormap and excluding alpha

    # Draw the points with colors based on importance
    viewer.display_points(
        points, color=colors, marker="point", scale=0.25 * max_length
    )


def draw_edge_uvgrids(solid, graph, viewer):
    edge_uvgrids = graph.edata["x"].view(-1, 6)
    points = []
    tangents = []
    importance_values = graph.edata["importance"].cpu().numpy()  # Assuming edge importance is stored in 'importance'
    norm = plt.Normalize(vmin=np.min(importance_values), vmax=np.max(importance_values))  # Normalize importance

    for idx in range(edge_uvgrids.shape[0]):
        points.append(edge_uvgrids[idx, :3].cpu().numpy())
        tangents.append(edge_uvgrids[idx, 3:6].cpu().numpy())

    points = np.asarray(points, dtype=np.float32)
    tangents = np.asarray(tangents, dtype=np.float32)

    bbox = solid.box()
    max_length = max(bbox.x_length(), bbox.y_length(), bbox.z_length())

    # Use a colormap to map the importance to colors
    colors = cm.viridis(norm(importance_values))[:, :3]  # Using viridis colormap and excluding alpha

    # Draw the points with colors based on importance
    viewer.display_points(points, color=colors, marker="point", scale=0.25 * max_length)


def visualize_uvgrids_and_graph(solid_file, graph_file):
    solid = load_step(solid_file)[0]
    graph = load_graphs(graph_file)[0][0]

    v = Viewer(backend="wx")
    # Draw the solid
    v.display(solid, transparency=0.5, color=(0.2, 0.2, 0.2))
    # Draw the face UV-grids with importance
    draw_face_uvgrids(solid, graph, viewer=v)
    # Draw the edge UV-grids with importance
    draw_edge_uvgrids(solid, graph, viewer=v)

    v.fit()
    v.show()
