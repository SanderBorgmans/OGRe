#! /usr/bin/python
import numpy as np, yaml, itertools, copy, os, warnings, glob
import matplotlib.path as Path
import matplotlib.pyplot as pt
import matplotlib.patches as patches

from scipy.spatial import Voronoi,voronoi_plot_2d

from molmod.io import load_chk
from molmod.units import *

__all__ = ['OGRe_Input']

warnings.filterwarnings('ignore')


def save2yaml(options,mode=None):
    d = vars(options)
    d = copy.deepcopy(d) # such that removing things from the dict do not affect options
    if mode is not None: d.update({'mode' : mode})
    if hasattr(options,'rvecs'):
        del d['rvecs']

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
        A class to create the input, which can be used to initialize
        the OGRe_Simulation instead of loading the text files
    '''
    def __init__(self,mode,**kwargs):
        '''
            **Simulations parameters**

            mode
                the potential mode

            kappas
                list of kappa values for each cv in fes_unit/cv_units**2

            spacings
                list of spacings for each cv in unit

            edges
                list of minimum and maximum bound for each cv in unit
                or already converted to dictionary with 'min' and 'max' as keys
                alternatively 'cof2d' to automatically generate a grid based on the unit cell

            cv_units
                list of molmod units as a string in which the cvs are expressed

            fes_units
                molmod unit as a string in which the free energy are expressed

            mdsteps
                number of md steps to execute

            runup
                steps to remove from the parsed trajectory data during post processing for equilibration

            h5steps
                saves the trajectory data every h5steps steps

            timestep
                time step for the MD integrator (in atomic units)

            temp
                temperature for the MD simulation (in atomic units)

            timecon_thermo
                timeconstant for the thermostat (in atomic units)

            pressure
                pressure for the MD simulation (in atomic units)

            timecon_baro
                timeconstant of the barostat (in atomic units)

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

        self.mode = mode

        # Assign the possible input values of kwargs
        for k,v in kwargs.items():
            setattr(self,k,v)

        # Sanity checks
        if self.mode=='application':
            if os.path.exists('init.chk'):
                chk_file = 'init.chk'
                chk = load_chk(chk_file)
            else:
                raise AssertionError('There was no structure file!')

            if len(glob.glob('pars*.txt'))==0:
                raise ValueError('No force field files found!')


        # SIMULATION PARAMETERS (other default values are assigned in sim/core.py)
        if not hasattr(self,'runup'):
            self.runup = 0 # no run up by default

        if not hasattr(self,'h5steps'):
            self.h5steps = 1 # save all data by default

        # Certain parameters should be evaluated to parse units
        for param in ['timestep', 'temp', 'press', 'timecon_thermo', 'timecon_baro']:
            if hasattr(self,param):
                val = getattr(self,param)
                val = eval(val) if isinstance(val,str) else val
                setattr(self, param, val)

        # Convert runup to h5step format (since the runup parameter is used when slicing)
        self.runup = self.runup//self.h5steps

        if not hasattr(self,'cv_units') or self.cv_units is None:
            self.cv_units = ['1']
        if len(self.cv_units)==1 and len(self.cv_units) < len(self.spacings):
            self.cv_units = self.cv_units*len(self.spacings)

        if not hasattr(self,'fes_unit') or self.fes_unit is None:   
            self.fes_unit = 'kjmol'

        # If cof2d setting True, calculate the edges instead of parsing them
        if hasattr(self,'cof2d') and self.cof2d:
            cv_units = get_cv_units(vars(self))
            assert cv_units[0]==cv_units[1]
            rvecs = chk['rvecs']/cv_units[0] # assume rvecs are in atomic units
            self.rvecs = rvecs # store rvecs in input class
            edges = get_edges(rvecs, self.spacings) 
            self.edges = {'min' : [float(edges[0]),float(edges[2])], 'max' : [float(edges[1]),float(edges[3])]}

        # Convert edges to dictionary for easy parsing
        if not isinstance(self.edges,dict):
            edges_dict = {'min':[self.edges[2*i]   for i,_ in enumerate(self.kappas)], 
                        'max':[self.edges[2*i+1] for i,_ in enumerate(self.kappas)]}
            self.edges = edges_dict

        # HYPERPARAMETERS
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

        if not hasattr(self,'HISTOGRAM_BIN_WIDTHS'):
            self.HISTOGRAM_BIN_WIDTHS = [spacing/self.MAX_LAYERS/2. for spacing in self.spacings]
            print('The histogram bin width for overlap calculation has been set to ', self.HISTOGRAM_BIN_WIDTHS)
    

        try:
            assert hasattr(self,'edges')
            assert hasattr(self,'spacings')
            assert hasattr(self,'kappas')
        except AssertionError:
            raise ValueError('You did not specify the required input parameters')

        try:
            if isinstance(self.edges,dict):
                assert len(self.edges['min'])==len(self.kappas) # check whether the number of edges is consistent with the number of kappas
            else:
                assert len(self.edges)==len(self.kappas)*2 # check whether the number of edges is consistent with the number of kappas
            assert len(self.kappas)==len(self.spacings)
        except AssertionError:
            raise ValueError('The number of specified edges/kappas/spacings was not consistent!')


        # Save all settings
        save2yaml(self,mode=self.mode)

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

def get_edges(rvecs,spacings):
    _,path = wigner_seitz_cell(np.array(rvecs)[0:2,0:2],plot=False)
    rows = [np.sort(np.hstack((np.arange(-sp,np.min(path.vertices[:,n])-sp,-sp), np.array([0]),  np.arange(sp, np.max(path.vertices[:,n])+sp, sp)))) for n,sp in enumerate(spacings)] # make square grid from -r to r (too big)
    return [np.min(rows[0]),np.max(rows[0]),np.min(rows[1]),np.max(rows[1])]

def get_rows(rvecs,spacings,plot):
    _,path = wigner_seitz_cell(np.array(rvecs)[0:2,0:2],plot=plot)
    r = np.max(np.linalg.norm(path.vertices,axis=-1))
    rows = [np.sort(np.hstack((np.arange(-sp,-r-sp,-sp), np.array([0]),  np.arange(sp, r+sp, sp)))) for sp in spacings] # make square grid from -r to r (too big)
    return rows,path

def write_grid(points,options,plot,path):
    with open('layer{0:0=2d}.txt'.format(0),'w') as f:
        f.write('layer,nr,cvs,kappas,type\n')
        for n,point in enumerate(points):
            f.write('{},{},{},{},{}\n'.format(0,n, '*'.join(['{:.8f}'.format(p) for p in point]), '*'.join(['{:.8e}'.format(k) for k in options.kappas]), 'new_node')) # use * as separator

     # Copy the layer00.txt file to run.txt
    from shutil import copyfile
    copyfile('layer00.txt', 'run.txt') # create initial run file

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

    if hasattr(options,'cof2d') and options.cof2d:
        rows, path = get_rows(options.rvecs,options.spacings,plot=options.plot)
        g = np.meshgrid(*rows,indexing='ij')
        mesh = np.vstack(map(np.ravel, g)).T
        mesh = mesh[path.contains_points(mesh,radius=np.min(options.spacings)*2.1)] # return points within path + fringe for better convergence of edges, change this to 2.1 for symmetric grid (it was 2.0 before)
        write_grid(mesh,options,plot,path)

    else:
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
