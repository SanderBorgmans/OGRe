#!/usr/bin/env python

import os,sys,copy,numpy as np
from yaff import *
from molmod import MolecularGraph

def get_indices(bonds, numbers):
    graph = MolecularGraph(bonds, numbers)
    indices = graph.independent_vertices
    return indices

def translate(P,v):
    trans_matrix= np.array( [[1,0,0,v[0]],[0,1,0,v[1]],[0,0,1,v[2]],[0,0,0,1]])
    todo = np.ones(4)
    todo[0] = P[0]
    todo[1] = P[1]
    todo[2] = P[2]
    return np.dot(trans_matrix,todo)

def get_cv(ff):
    assert os.path.exists('atom_types')
    with open('atom_types','r') as f:
        atypes = set([line.strip() for line in f.readlines()])
    ls = get_indices(ff.system.bonds, ff.system.numbers)
    l1 = ls[0]
    l2 = ls[1]
    layers = [np.array([l for l in layer if ff.system.get_ffatype(l) in atypes],dtype=int) for layer in [l1,l2]]
    return [colvar.COMProjection(groups=layers)]

def adapt_structure(ff,cv):
    cv_object = get_cv(ff)[0]

    # Adapt stucture for cv
    pos = copy.copy(ff.system.pos)
    ls = get_indices(ff.system.bonds, ff.system.numbers)
    for idx in ls[1]:
        pos[idx] = translate(pos[idx],[*cv])[0:3]
    ff.update_pos(pos)
    cell_symmetrize(ff)

    try:
        assert np.array_equal(np.array(cv)[:2],cv_object.get_cv_value(ff.system)[:2])
    except AssertionError:
        print(cv, cv_object.get_cv_value(ff.system))
        sys.exit(1)
