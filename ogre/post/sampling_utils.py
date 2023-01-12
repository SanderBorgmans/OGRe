
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
    try:
        points = np.round(np.array([node.loc for node in nodes]),DECIMALS)
    except TypeError:
        print([node.loc for node in nodes])
        print([node.type for node in nodes])
        print()
        raise TypeError
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
    