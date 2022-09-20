#! /usr/bin/python
import numpy as np, yaml, itertools, copy, os, warnings
import matplotlib.path as Path
import matplotlib.pyplot as pt
import matplotlib.patches as patches

from scipy.spatial import Voronoi,voronoi_plot_2d

from molmod.io import load_chk
from molmod.units import *

__all__ = ['OGRe_Input']

warnings.filterwarnings('ignore')

def save2yaml(options):
    d = vars(options)
    if os.path.exists('data.yml'):
        with open('data.yml','r') as yamlfile:
            cyaml = yaml.safe_load(yamlfile) # Note the safe_load
        if cyaml is None:
            cyaml = d
        else:
            cyaml.update(d)
        with open('data.yml','w') as yamlfile:
            yaml.safe_dump(cyaml, yamlfile) # Also note the safe_dump
    else:
        with open('data.yml','w') as yamlfile:
            yaml.safe_dump(d, yamlfile) # Also note the safe_dump


class OGRe_Input(object):
    '''
        A class to create the input for your prefered simulation engine
    '''
    def __init__(self,kwargs):
        '''
            **Simulations parameters**

            kappas
                list of kappa values for each cv in fes_unit/cv_units**2

            spacings
                list of spacings for each cv in unit

            edges
                list of minimum and maximum bound for each cv in unit

            cv_units
                list of molmod units as a string in which the cvs are expressed

            fes_units
                molmod unit as a string in which the free energy are expressed

            runup
                steps to remove from the parsed trajectory data during post processing for equilibration

            **Hyperparameters** - see readme

                CONFINEMENT_THR [default=0.3]
                OVERLAP_THR [default=0.3]
                KAPPA_GROWTH_FACTOR [default=2]
                MAX_LAYERS [default=1]
                MAX_KAPPA [default=None]
                HISTOGRAM_BIN_WIDTHS [default=spacings/MAX_LAYERS/2]

            plot
                plot the cv grid, defaults to True

        '''

        # Assign the possible input values of kwargs
        for k,v in kwargs.items():
            setattr(self,k,v)

        # Sanity checks
        try:
            assert self.edges is not None
            assert self.spacings is not None
            assert self.kappas is not None
        except AssertionError:
            raise ValueError('You did not specify the required input parameters')

        try:    
            assert len(self.edges)==len(self.kappas)*2 # check whether the number of edges is consistent with the number of kappas
            assert len(self.kappas)==len(self.spacings)
        except AssertionError:
            raise ValueError('The number of specified edges/kappas/spacings was not consistent!')

        if self.cv_units is None:
            self.cv_units = ['1']
        if len(self.cv_units)==1 and len(self.cv_units) < len(self.spacings):
            self.cv_units = self.cv_units*len(self.spacings)

        # Convert edges to dictionary for easy parsing
        edges_dict = {'min':[self.edges[2*i]   for i,_ in enumerate(self.kappas)], 
                      'max':[self.edges[2*i+1] for i,_ in enumerate(self.kappas)]}
        self.edges = edges_dict
        
        # Set default values
        if not hasattr(self,'plot'):
            self.plot = True

        if not hasattr(self,'CONFINEMENT_THR'):
            self.CONFINEMENT_THR = 0.3

        if not hasattr(self,'OVERLAP_THR'):
            self.OVERLAP_THR = 0.3

        if not hasattr(self,'KAPPA_GROWTH_FACTOR'):
            self.KAPPA_GROWTH_FACTOR = 2

        if not hasattr(self,'MAX_LAYERS'):
            self.MAX_LAYERS = 1

        if not hasattr(self,'HISTOGRAM_BIN_WIDTH'):
            self.HISTOGRAM_BIN_WIDTHS = [spacing/self.MAX_LAYERS/2. for spacing in self.spacings]
            print('The histogram bin width for overlap calculation has been set to ', self.HISTOGRAM_BIN_WIDTHS)
    
        # Save all settings
        save2yaml(self)

    def make_grid(self):
        self.grid = make_grid(self)


def sort_vertices(vertices):
    # sort according to angle, and add first vertex as last one to ensure closure
    com_vertices = vertices - np.average(vertices,axis=0)
    angles = np.arctan2(com_vertices[:,1],com_vertices[:,0])
    sorted_v = vertices[angles.argsort()]
    return np.concatenate((sorted_v,[sorted_v[0]]),axis=0)

def wigner_seitz_cell(vecs,plot=True):
    assert vecs.shape[0]==2
    # make wigner seitz cell boundary
    images = np.array([sum(n * vec for n, vec in zip(ns, vecs)) for ns in itertools.product([-1,0,1],repeat=2)]) # 2 dimensions
    images = images[np.where(np.linalg.norm(images,axis=-1)!=np.linalg.norm(images,axis=-1).max())] # get nearest neighbors
    vor = Voronoi(images)
    if plot:
        fig = voronoi_plot_2d(vor)
        pt.show()
        pt.close()
    return vor,Path.Path(sort_vertices(vor.vertices), closed=True)  # make sure that ordening is right

def make_path(min,max):
    if len(min) == 1:
        vertices = np.array([[min[0],-1],[min[0],1],[max[0],-1],[max[0],1]])
    elif len(min) == 2:
        vertices = np.array([[min[0],min[1]],[min[0],max[1]],[max[0],min[1]],[max[0],max[1]]])
    else:
        return None
    return Path.Path(sort_vertices(vertices), closed=True)

def get_edges(rvecs):
    _,path = wigner_seitz_cell(rvecs[0:2,0:2],False)
    vertices = path.vertices
    return [np.min(vertices[:,0]),np.max(vertices[:,0]),np.min(vertices[:,1]),np.max(vertices[:,1])]

def get_rows(rvecs,spacings):
    _,path = wigner_seitz_cell(rvecs[0:2,0:2])
    r = np.max(np.linalg.norm(path.vertices,axis=-1))
    rows = [np.sort(np.hstack((np.arange(-sp,-r-sp,-sp), np.array([0]),  np.arange(sp, r+sp, sp)))) for sp in spacings] # make square grid from -r to r (too big)
    return rows,path

def write_grid(points,options,plot,path):
    with open('grid{0:0=2d}.txt'.format(0),'w') as f:
        f.write('grid,nr,cvs,kappas\n')
        for n,point in enumerate(points):
            f.write('{},{},{},{}\n'.format(0,n, '*'.join(['{:.8f}'.format(p) for p in point]), '*'.join(['{:.8e}'.format(k) for k in options.kappas]))) # use * as separator

    if plot:
        if points.shape[1] > 2:
            raise NotImplementedError
        elif points.shape[1] == 1:
            pt.figure(figsize=(10,2))
            pt.scatter(points[:,0],np.zeros_like(points[:,0]))
            path.vertices = path.vertices
            patch = patches.PathPatch(path, lw=2, alpha=0.1)
            ax = pt.gca()
            ax.add_patch(patch)
            pt.yticks([])
            pt.show()
        elif points.shape[1] == 2:
            pt.figure(figsize=(8,8))
            pt.scatter(points[:,0],points[:,1])
            path.vertices = path.vertices
            patch = patches.PathPatch(path, lw=2, alpha=0.1)
            ax = pt.gca()
            ax.add_patch(patch)
            pt.show()

def make_grid(options):
    edges = options.edges
    plot = options.plot
    assert edges is not None
    min = np.asarray(edges['min'])
    max = np.asarray(edges['max'])
    assert((min<max).all())
    assert len(min) == len(max)

    assert len(min) == len(options.spacings)
    assert len(min) == len(options.kappas)
    options.spacings = np.array(options.spacings)

    arrs = [np.arange(min[n],max[n]+options.spacings[n],options.spacings[n]) for n,_ in enumerate(min)]
    g = np.meshgrid(*arrs,indexing='ij')
    mesh = np.vstack(map(np.ravel, g)).T
    write_grid(mesh,options,plot,make_path(min,max))

def get_cv_units(data):
    if 'cv_units' in data:
        if not isinstance(data['cv_units'], list):
            units = [data['cv_units']] * len(data['spacings'])
        else:
            units = data['cv_units']
        assert len(units)==len(data['spacings'])

        units = np.array([eval(unit) if unit is not None else 1.0 for unit in units])
    else:
        units = np.ones(len(data['spacings']))
    return units
