#! /usr/bin/python
import numpy as np

import matplotlib.pyplot as pt
import matplotlib.patches as mpatches

from scipy.spatial import KDTree

from ogre.input.utils import wigner_seitz_cell

DECIMALS = 5
precision = 10**(-DECIMALS)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_dev(ax, node, traj, steps):
    if len(node.loc) == 1:
        ax.scatter(traj[:],np.zeros_like(traj[:]),s=0.1, c='r')
        ax.scatter([node.loc[0]],[0],s=20.0,marker='x',c='r')
        ax.vlines(((node.loc[0]+ np.array([steps[0],-steps[0]]))),-0.5,0.5)
        ax.set_xlim([(node.loc[0]-2*steps[0]),(node.loc[0]+2*steps[0])])
        ax.set_ylim([-0.5,0.5])
    elif len(node.loc) == 2:
        _,path = wigner_seitz_cell(np.diag(steps*2),False)
        path.vertices += node.loc
        patch = mpatches.PathPatch(path, facecolor="none", lw=1)

        ax.scatter(traj[:,0],traj[:,1],s=0.1, c='r')
        ax.scatter([node.loc[0]],[node.loc[1]],s=20.0,marker='x',c='r')
        ax.add_patch(patch)
        ax.set_xlim([(node.loc[0]-2*steps[0]),(node.loc[0]+2*steps[0])])
        ax.set_ylim([(node.loc[1]-2*steps[1]),(node.loc[1]+2*steps[1])])
    else:
        raise NotImplementedError("Can't make deviation plots in more than 2 dimensions.")


def plot_overlap(ax, node1, node2, t1, trajs2, steps):
    if len(node1.loc) == 1:
        ax.scatter([node1.loc[0]],[0],s=30.0,marker='x',c='r',label='t1_loc')
        ax.scatter([node2.loc[0]],[0],s=30.0,marker='x',c='b',label='t2_loc')
        if len(trajs2)>1:
            for t2 in trajs2:
                pt.scatter(t2[0,0],0,s=40,marker='x', c='g')#, label='t2')
        pt.scatter(t1,np.zeros_like(t1),s=0.1, c='r')#, label='t1')
        for t2 in trajs2:
            pt.scatter(t2,np.zeros_like(t2),s=0.1, c='b')#, label='t2')
        pt.xlim([(node1.loc[0]-4*steps[0]),(node1.loc[0]+4*steps[0])])
    elif len(node1.loc) == 2:
        ax.scatter([node1.loc[0]],[node1.loc[1]],s=30.0,marker='x',c='r',label='t1_loc')
        ax.scatter([node2.loc[0]],[node2.loc[1]],s=30.0,marker='x',c='b',label='t2_loc')
        if len(trajs2)>1:
            for t2 in trajs2:
                ax.scatter(t2[0,0],t2[0,1],s=40,marker='x', c='g')#, label='t2')
        ax.scatter(t1[:,0],t1[:,1],s=0.1, c='r')#, label='t1')
        for t2 in trajs2:
            ax.scatter(t2[:,0],t2[:,1],s=0.1, c='b')#, label='t2')
        ax.set_xlim([(node1.loc[0]-4*steps[0]),(node1.loc[0]+4*steps[0])])
        ax.set_ylim([(node1.loc[1]-4*steps[1]),(node1.loc[1]+4*steps[1])])
    else:
        raise NotImplementedError("Can't make overlap plots in more than 2 dimensions.")


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
