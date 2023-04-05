
#! /usr/bin/python
import numpy as np
from scipy.spatial import cKDTree as KDTree # identical to KDTree but a lot faster before v1.6.0 

DECIMALS = 5
precision = 10**(-DECIMALS)

def norm(point,spacings):
    # check whether point lies within ellipsoid
    return np.sum((point/spacings)**2)


def find_neighbours(steps,nodes):
    """
        Find the neighbours for each node
    """
    points = np.round(np.array([node.loc for node in nodes]),DECIMALS)

    tree = KDTree(points)
    neighbours = []
    for n,node in enumerate(nodes):
        point = node.loc
        ind = []
        for i in range(point.shape[0]):
            step = np.zeros_like(steps)
            step[i] = steps[i]
            _, n1 = tree.query(point + step,1)
            _, n2 = tree.query(point - step,1)
            ind.append(n1)
            ind.append(n2)

        # if the index corresponds to the point itself it should be omitted (edge/corner point)
        ind = [i for i in ind if not i==n and norm(point-nodes[i].loc,steps*1.1)<1.] # safety factor 1.1
        neighbours.append(ind)
    return neighbours


def make_unique_between(steps,ref_points,test_points, get_mask=False):
    nrefpoints = np.round(np.array([rp/steps for rp in ref_points]),DECIMALS) #normalized points
    ntestpoints = np.round(np.array([tp/steps for tp in test_points]),DECIMALS) #normalized points

    mask = np.ones(len(ntestpoints),dtype=bool)
    tree = KDTree(nrefpoints)

    for n,point in enumerate(ntestpoints):
        idx = tree.query_ball_point(point,0.5)
        if len(idx)>0:
            mask[n] = 0
    if get_mask:
        return mask
    return test_points[mask]


def make_unique(steps,points,get_mask=False):
    npoints = np.round(np.array([p/steps for p in points]),DECIMALS) #normalized points
    mask = np.ones(len(npoints),dtype=bool)
    tree = KDTree(npoints)

    for n,point in enumerate(npoints):
        idx = tree.query_ball_point(point,0.5)
        if all([mask[i] for i in idx]) and len(idx)>1:
            mask[idx[1:]] = 0
    if get_mask:
        return mask
    return points[mask]


def find_node_index(tree,point):
    ds,idx = tree.query(np.round(point,DECIMALS),1) # Find the point
    return ds<precision, idx
    

def get_grid_edges(grid):
    """
    Calculate the edges for each grid point in a grid of coordinates.

    Args:
        grid (ndarray): An M-dimensional grid of coordinates with shape (N_1, N_2, ..., N_M, M).

    Returns:
        tuple: A tuple of M ndarrays, where each ndarray represents the one-dimensional
        grid edges along that dimension.
    """
    edges = []
    width_list = []
    for dim in range(grid.shape[-1]):
        # Calculate the edge widths along this dimension
        widths = np.diff(grid[..., dim], axis=dim).ravel()
        width = widths[0]
        # Assume the grid is uniform
        assert np.allclose(widths,width)

        # Get a single row of the values along this dimension
        fixed_dims = (0,) * dim + np.index_exp[:] + (0,) * (grid.shape[-1] - dim - 1) + (dim,)
        row = grid[fixed_dims]

        # Calculate edges
        dim_edges = np.append(row - width/2.,row[-1]+width/2.)
        edges.append(dim_edges)
        width_list.append(width)

    return tuple(edges), width_list