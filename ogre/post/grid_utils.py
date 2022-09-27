#! /usr/bin/python
import os,time,h5py,glob,pickle,sys,warnings
import numpy as np
import matplotlib.pyplot as pt

from molmod.io import load_chk
from molmod.units import *
from ogre.input.utils import wigner_seitz_cell
from ogre.post.sampling import ReferenceNode, Node

warnings.filterwarnings('ignore')

def load_potential_file(file_loc):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module",file_loc)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_compressed_trajs(compressed_fn,number,grids,trajs,dkappas,identities):
    with h5py.File(compressed_fn,'r') as f:
        grids[number]   = np.array(f['grids'])
        trajs[number]   = np.array(f['trajs'])
        dkappas[number] = np.array(f['dkappas'])
        identities[number] = [tuple(identity) for identity in f['identities']]

def load_trajs(gn,data,restart,grids,trajs,dkappas,identities):
    grid = np.genfromtxt(gn, delimiter=',',dtype=None,skip_header=1)

    if grid.size==1: # problem with single line files
        grid = np.array([grid])

    for point in grid:
        # Define identity
        gnr = int(point[0])
        nr = int(point[1])
        identity = (gnr,nr)

        # Multi load: allows the loading of both compressed trajectories and non-compressed trajectories
        # This could occur in cases that the initial grid edges would be extended
        if 'multi_load' in data:
            multi_load = data['multi_load']
            # In this case some data points may have already been loaded
            if multi_load and identity in identities[gnr]:
                continue # skip this point

        # Define point
        if isinstance(point[2],float):
            cvs = np.array([point[2]])
        else:
            cvs = np.array([float(p) for p in point[2].decode().split('*')])

        # Define kappas
        if isinstance(point[3],float):
            kappas = np.array([point[3]])
        else:
            kappas = np.array([float(k) for k in point[3].decode().split('*')])
        try:
            with h5py.File('trajs/traj_{}_{}.h5'.format(gnr,nr),'r') as f:
                tr = f['trajectory/cv_values'][:].reshape((1,-1,len(kappas)))

                if not gnr in grids.keys() or not gnr in trajs.keys():
                    grids[gnr]   = cvs
                    trajs[gnr]   = tr
                    dkappas[gnr] = kappas
                    identities[gnr] = [identity]
                else:
                    grids[gnr]   = np.vstack((grids[gnr], cvs))
                    trajs[gnr]   = np.vstack((trajs[gnr], tr))
                    dkappas[gnr] = np.vstack((dkappas[gnr], kappas))
                    identities[gnr].append(identity)

        except OSError:
            restart += '{},{},{},{}\n'.format(identity[0],identity[1],"*".join(['{:.8f}'.format(p) for p in cvs]),"*".join(['{:.8e}'.format(k) for k in kappas]))

    if not restart == "":
        gr = open('grid_restart.txt','w')
        gr.write('grid,nr,cvs,kappas\n')
        gr.write(restart)
        gr.close()
        raise ValueError("Some simulations have to be restarted!")


def load_grid(data,index=None,verbose=True):
    """
        Create grid if it does not yet exist
    """
    # Load all grids and divide in sub grids based on gridnr
    grids = {}
    trajs = {}
    dkappas = {}
    identities = {}

    restart = ""

    multi_load = False
    if 'multi_load' in data:
        multi_load = data['multi_load']

    init = time.time()
    if not os.path.exists('trajs.pkl') or not os.path.exists('grids.pkl') or not os.path.exists('kappas.pkl'):
        grid_names = sorted(glob.glob('grid[0-9][0-9].txt'), key = lambda gn: int(gn.split('.')[-2].split('grid')[-1]))
        numbers = [int(gn.split('.')[-2].split('grid')[-1]) for gn in grid_names]
        if index is not None and max(numbers)<index:
            print('Max index reached: {} < {}'.format(max(grids.keys()),index))
            sys.exit()

        for n,gn in enumerate(grid_names):
            number = numbers[n]

            # If we are extending grids and already have some iterations, this will not take the subsequent iterations into account
            if 'MAX_LAYERS' in data and not number < data['MAX_LAYERS']:
                continue

            compressed_fn = 'trajs/compressed_{}.h5'.format(number)
            compressed = os.path.exists(compressed_fn)
            if multi_load:
                # Load both the compressed part and new trajectories, this is convenient when expanding grid
                print("Loading both compressed file and trajectory files for grid {}.".format(number))
                assert compressed
                load_compressed_trajs(compressed_fn,number,grids,trajs,dkappas,identities)
                load_trajs(gn,data,restart,grids,trajs,dkappas,identities)
            else:
                if compressed:
                    print("Loading compressed file for grid {}.".format(number))
                    load_compressed_trajs(compressed_fn,number,grids,trajs,dkappas,identities)
                else:
                    print("Loading trajectory files for grid {}.".format(number))
                    load_trajs(gn,data,restart,grids,trajs,dkappas,identities)

        with open('trajs.pkl','wb') as fp:
            pickle.dump(trajs,fp)
        with open('grids.pkl','wb') as fp:
            pickle.dump(grids,fp)
        with open('kappas.pkl','wb') as fp:
            pickle.dump(dkappas,fp)
        with open('identities.pkl','wb') as fp:
            pickle.dump(identities,fp)
    else:
        with open('trajs.pkl','rb') as fp:
            trajs = pickle.load(fp)
        with open('grids.pkl','rb') as fp:
            grids = pickle.load(fp)
        with open('kappas.pkl','rb') as fp:
            dkappas = pickle.load(fp)
        with open('identities.pkl','rb') as fp:
            identities = pickle.load(fp)

    if verbose:
        print("Loading trajectories took {} seconds.".format(time.time()-init))
    if index is not None:
        if index > max(grids.keys()):
            print('Max index reached: {} < {}'.format(max(grids.keys()),index))
            sys.exit(0)

        grids   = { k:v for k,v in grids.items() if k<index+1 }
        trajs   = { k:v for k,v in trajs.items() if k<index+1 }
        dkappas = { k:v for k,v in dkappas.items() if k<index+1 }
        identities = { k:v for k,v in identities.items() if k<index+1 }

    return grids, trajs, dkappas, identities

def write_grid(n,data,grid,test=False):
    if test: # if test the user does not want any files to be created, changed, or removed
        return
    if 'MAX_LAYERS' in data and n+1==data['MAX_LAYERS']:
        #if len(grid.finer_nodes)>0:
        print('MAX_LAYERS reached.')
    else:
        fname = 'grid{0:0=2d}.txt'.format(n+1)
        # Check if grid file already exists, if it does, append new grid points (assume only grid points are added with refinement, not removed)
        if os.path.exists(fname):
            # Read the existing grid points
            with open(fname,'r') as f:
                lines = f.readlines()
                lines = lines[1:] # skipheader

            # Assume location alone is enough to identify points
            ids_existing = []
            locs_existing = []
            kappas_existing = []
            for line in lines:
                identity = int(line.split(',')[1])
                loc = tuple([float(l) for l in line.split(',')[2].split('*')])
                kappa = tuple([float(k) for k in line.split(',')[3].split('*')])
                ids_existing.append(identity)
                locs_existing.append(loc)
                kappas_existing.append(kappa)

            # Sanity check
            try:
                assert max(ids_existing)+1==len(lines)
            except AssertionError:
                raise ValueError('An error occured, and some identities are not unique')
            except ValueError:
                pass # no existing ids

            locs_grid = {}
            for node in grid.finer_nodes:
                locs_grid[tuple(node.loc)] = node

        
            with open(fname,'a') as f:
                idx=0
                for k,node in locs_grid.items():
                    if k in locs_existing: # check if kappa value remained unaltered
                        try:
                            assert tuple(node.kappas)==kappas_existing[locs_existing.index(k)]
                        except AssertionError:
                            continue
                    else: # write the new grid point
                        f.write('{},{},{},{}\n'.format(n+1,len(lines)+idx,"*".join(['{:.8f}'.format(p) for p in node.loc]),"*".join(['{:.8e}'.format(kappa) for kappa in node.kappas])))
                        idx+=1

        else: # else make the grid file
            with open(fname,'w') as f:
                f.write('grid,nr,cvs,kappas\n')
                idx = 0
                for node in grid.finer_nodes:
                    f.write('{},{},{},{}\n'.format(n+1,idx,"*".join(['{:.8f}'.format(p) for p in node.loc]),"*".join(['{:.8e}'.format(kappa) for kappa in node.kappas])))
                    idx += 1

    # Add the tagged virtual reference nodes to the previous grid as virtual finer nodes which are then taken into account when loading all trajs
    if not grid.major_grid is None:
        # Read the grid files first to check idx value, and whether no points are already included
        fname = 'grid{0:0=2d}.txt'.format(n)
        with open(fname,'r') as f:
            lines = f.readlines()
            lines = lines[1:] # skipheader

        ids_existing = []
        locs_existing = []
        kappas_existing = []
        for line in lines:
            identity = int(line.split(',')[1])
            loc = tuple([float(l) for l in line.split(',')[2].split('*')])
            kappa = tuple([float(k) for k in line.split(',')[3].split('*')])
            ids_existing.append(identity)
            locs_existing.append(loc)
            kappas_existing.append(kappa)

        idx = max(ids_existing) + 1

        with open(fname,'a') as f:
            for node in grid.realized_virtual_reference_nodes:
                if tuple(node.loc) not in locs_existing:
                    f.write('{},{},{},{}\n'.format(n,idx,"*".join(['{:.8f}'.format(p) for p in node.loc]),"*".join(['{:.8e}'.format(kappa) for kappa in node.kappas])))
                    idx += 1
                else:
                    print('This node already existed! Skipping ...')

    # Replace the grid point lines in the previous grids if their confinement was not sufficient
    nodes_dict = {}
    for node in grid.deviant_nodes:
        if node.identity[0] not in nodes_dict:
            nodes_dict[node.identity[0]] = [node]
        else:
            nodes_dict[node.identity[0]].append(node)

    for k,v in nodes_dict.items():
        with open('grid{0:0=2d}.txt'.format(k),'r') as f:
            lines = f.readlines()
        for node in v:
            lines[node.identity[1]+1] = '{},{},{},{}\n'.format(node.identity[0],node.identity[1],"*".join(['{:.8f}'.format(p) for p in node.loc]),"*".join(['{:.8e}'.format(k) for k in node.kappas]))
            # Remove deviant trajectory
            os.remove('trajs/traj_{}_{}.h5'.format(node.identity[0],node.identity[1]))

        with open('grid{0:0=2d}.txt'.format(k),'w') as f:
            f.writelines(lines)


def plot_grid(n,data,grid):
    fig = pt.figure('grid')
    ax = fig.gca()
    # Plot previous grids
    tmp = grid
    while not tmp.major_grid is None:
        tmp = tmp.major_grid
        tmp_grid = np.array([node.loc for node in tmp.nodes])
        if tmp_grid.shape[1] == 1:
            ax.scatter(tmp_grid[:,0], np.zeros_like(tmp_grid[:,0]), s=0.1, color='gray')
            ax.scatter(np.array([node.loc for node in tmp.nodes if isinstance(node,Node) and node.extreme_kappa]),
         np.zeros_like(np.array([node.loc for node in tmp.nodes if isinstance(node,Node) and node.extreme_kappa])), s=0.1, marker='x', color='gray')
        elif tmp_grid.shape[1] == 2:
            ax.scatter(tmp_grid[:,0],tmp_grid[:,1], s=0.1, color='gray')
            ax.scatter(np.array([node.loc[0] for node in tmp.nodes if isinstance(node,Node) and node.extreme_kappa]),
                       np.array([node.loc[1] for node in tmp.nodes if isinstance(node,Node) and node.extreme_kappa]), s=0.1, marker='x', color='gray')
        else:
            break

    # Plot current grid and refinement
    grid_grid = np.array([node.loc for node in grid.nodes])
    if grid_grid.shape[1] == 1:
        fig.set_size_inches(((n+1)*4,1))
        ax.scatter(grid_grid[:,0],np.zeros_like(grid_grid[:,0]), s=0.2, color='k')
        ax.scatter(np.array([node.loc[0] for node in grid.finer_nodes if not node.reference]),
     np.zeros_like(np.array([node.loc[0] for node in grid.finer_nodes if not node.reference])), s=0.05, color='r')
        ax.scatter(np.array([node.loc[0] for node in grid.finer_nodes if node.reference]),
     np.zeros_like(np.array([node.loc[0] for node in grid.finer_nodes if node.reference])), s=0.05, color='b',marker='x', alpha=0.5)
        ax.scatter(np.array([node.loc[0] for node in grid.deviant_nodes]),
     np.zeros_like(np.array([node.loc[0] for node in grid.deviant_nodes])), s=5., color='b',marker='x')
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if not rnode.real]),
     np.zeros_like(np.array([rnode.loc[0] for rnode in grid.reference_nodes if not rnode.real])), s=0.05, color='g', marker='x')
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and rnode.to_realize]),
     np.zeros_like(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and rnode.to_realize])), s=0.05, color='orange', marker='x', alpha=0.5)
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and not rnode.to_realize]),
     np.zeros_like(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and not rnode.to_realize])), s=0.05, color='g', marker='x', alpha=0.5)
    elif grid_grid.shape[1] == 2:
        fig.set_size_inches(((n+1)*2,(n+1)*2))
        ax.scatter(grid_grid[:,0],grid_grid[:,1], s=0.2, color='k')
        ax.scatter(np.array([node.loc[0] for node in grid.finer_nodes]),
                   np.array([node.loc[1] for node in grid.finer_nodes]), s=0.05, color='r')
        ax.scatter(np.array([node.loc[0] for node in grid.deviant_nodes]),
                   np.array([node.loc[1] for node in grid.deviant_nodes]), s=0.075, color='b', marker='x')
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if not rnode.real]),
                   np.array([rnode.loc[1] for rnode in grid.reference_nodes if not rnode.real]), s=0.05, color='g', marker='x')
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and rnode.to_realize]),
                   np.array([rnode.loc[1] for rnode in grid.reference_nodes if rnode.real and rnode.to_realize]), s=0.05, color='orange', marker='x', alpha=0.5)
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and not rnode.to_realize]),
                   np.array([rnode.loc[1] for rnode in grid.reference_nodes if rnode.real and not rnode.to_realize]), s=0.05, color='g', marker='x', alpha=0.5)
    else:
        pt.close('grid')

    fig.savefig('grid_points_{}.pdf'.format(n),bbox_inches='tight')
    pt.close('grid')

def write_fes(data,grid,fes,index,suffix=None,fes_err=None):
    # Grid is expressed in units and does not need to be converted
    fes_data = np.concatenate((grid,fes.reshape(*fes.shape,1)),axis=-1)
    fes_data = fes_data.reshape((-1,fes_data.shape[-1]))
    fname = 'fes'
    if index is not None:
        fname += '_{}'.format(index)
    if suffix is not None:
        fname += '_{}'.format(suffix)

    np.savetxt(fname+'.dat', fes_data, fmt='%.9e', delimiter='\t')

    if fes_err is not None and not np.isnan(fes_err).all():
        fes_err_data = np.concatenate((grid,fes_err[0].reshape(*fes_err[0].shape,1),fes_err[1].reshape(*fes_err[1].shape,1)),axis=-1)
        fes_err_data = fes_err_data.reshape((-1,fes_err_data.shape[-1]))
        fname = 'fes_err'
        if index is not None:
            fname += '_{}'.format(index)
        if suffix is not None:
            fname += '_{}'.format(suffix)

        np.savetxt(fname+'.dat', fes_err_data, fmt='%.9e', delimiter='\t')


def plot_fes(data,grid,fes,index,avg_window=1,suffix=None,fes_err=None):
    # Grid is expressed in units and does not need to be converted
    fes-= np.nanmin(fes)

    fig = pt.figure('fes',figsize = (6,6))
    ax = fig.gca()
    if grid.ndim == 2:
        N = avg_window
        avg_fes = np.convolve(fes, np.ones((N,))/N, mode='valid')
        avg_grid = np.convolve(grid[:,0], np.ones((N,))/N, mode='valid')

        try:
            potential = load_potential_file('./potential.py').Potential
            ax.plot(avg_grid,potential.eval(avg_grid)-np.min(potential.eval(avg_grid)),'--')
        except (FileNotFoundError,NotImplementedError):
            if data['mode'] == 'analytical':
                print('There was no potential.py file or eval method available for comparison with calculated FES')
        ax.plot(avg_grid,avg_fes-np.nanmin(avg_fes))

        if fes_err is not None and not np.isnan(fes_err).all():
            ax.fill_between(grid[:,0], fes_err[0], fes_err[1], alpha=0.33)

    elif grid.ndim == 3:
        cmap = 'viridis_r'
        im = ax.contourf(grid[:,:,0],grid[:,:,1],fes,31,cmap=cmap)
        fig.colorbar(im)
    else:
        raise NotImplementedError('Plotting the FES on a grid with more than 2 dimensions is not implemented.')

    fname = 'us_plot'
    if index is not None:
        fname += '_{}'.format(index)
    if suffix is not None:
        fname += '_{}'.format(suffix)

    fig.savefig(fname+'.pdf',bbox_inches='tight')
    pt.close('fes')


def cut_edges(data,grid):
    edges = data['edges']
    spacings = data['spacings']
    for i,_ in enumerate(spacings):
        grid.finer_nodes = [node for node in grid.finer_nodes if node.loc[i] < edges['max'][i]+spacings[i] and node.loc[i] > edges['min'][i]-spacings[i]]
        grid.reference_nodes = [rnode for rnode in grid.reference_nodes if rnode.loc[i] < edges['max'][i]+spacings[i] and rnode.loc[i] > edges['min'][i]-spacings[i]]


def write_colvars(filename,rtrajs,rgrids,rkappas,verbose=True):
    if not os.path.exists('colvars/'):
        os.mkdir('colvars')

    with open(filename,'w') as g:
        for n,traj in enumerate(rtrajs):
            # Define point
            cvs = rgrids[n]
            # Define kappa
            kappas = rkappas[n]
            if verbose:
                print("{} | {} - {}".format(n," ".join(["{: 2.2f}".format(cv) for cv in cvs])," ".join(["{: 2.2f}".format(k) for k in kappas])))

            # Create a 1D time series of the collective variable
            t = np.arange(0,len(traj)).reshape(-1,1)

            # Save this time series in a file named colvar
            np.savetxt('colvars/colvar_{}'.format(n), np.hstack((t, traj)))

            # Write the value of the collective variable and the harmonic spring constant
            g.write('colvars/colvar_{}\t'.format(n) + "\t".join([str(cv) for cv in cvs]) + '\t' + "\t".join([str(kappa) for kappa in kappas]) + '\n')
            
