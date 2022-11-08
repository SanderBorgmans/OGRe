#! /usr/bin/python
import os,time,h5py,glob,pickle,sys,warnings
import numpy as np
import matplotlib.pyplot as pt

from molmod.io import load_chk
from molmod.units import *
from ogre.input.utils import wigner_seitz_cell
from ogre.post.sampling import ReferenceNode, Node
from ogre.post.sampling_utils import precision

warnings.filterwarnings('ignore')

def load_potential_file(file_loc):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module",file_loc)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_compressed_trajs(compressed_fn,number,grids,trajs,dkappas,identities,types):
    with h5py.File(compressed_fn,'r') as f:
        grids[number]   = np.array(f['grids'])
        trajs[number]   = np.array(f['trajs'])
        dkappas[number] = np.array(f['dkappas'])
        identities[number] = [tuple(identity) for identity in f['identities']]
        types[number] = [t for t in f['types']]

def load_trajs(gn,data,restart,grids,trajs,dkappas,identities,types):
    grid = np.genfromtxt(gn, delimiter=',',dtype=None,skip_header=1)

    if grid.size==1: # problem with single line files
        grid = np.array([grid])

    for point in grid:
        # Define identity
        gnr = int(point[0])
        nr = int(point[1])
        identity = (gnr,nr)
        dtype = point[4].decode()

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
                    types[gnr] = [dtype]
                else:
                    grids[gnr]   = np.vstack((grids[gnr], cvs))
                    trajs[gnr]   = np.vstack((trajs[gnr], tr))
                    dkappas[gnr] = np.vstack((dkappas[gnr], kappas))
                    identities[gnr].append(identity)
                    types[gnr].append(dtype)

        except OSError:
            restart += '{},{},{},{}\n'.format(identity[0],identity[1],"*".join(['{:.8f}'.format(p) for p in cvs]),"*".join(['{:.8e}'.format(k) for k in kappas]),dtype)

    if len(restart)>0:
        gr = open('grid_restart.txt','w')
        gr.write('grid,nr,cvs,kappas,type\n')
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
    types = {}

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
                load_compressed_trajs(compressed_fn,number,grids,trajs,dkappas,identities,types)
                load_trajs(gn,data,restart,grids,trajs,dkappas,identities,types)
            else:
                if compressed:
                    print("Loading compressed file for grid {}.".format(number))
                    load_compressed_trajs(compressed_fn,number,grids,trajs,dkappas,identities,types)
                else:
                    print("Loading trajectory files for grid {}.".format(number))
                    load_trajs(gn,data,restart,grids,trajs,dkappas,identities,types)

        with open('trajs.pkl','wb') as fp:
            pickle.dump(trajs,fp)
        with open('grids.pkl','wb') as fp:
            pickle.dump(grids,fp)
        with open('kappas.pkl','wb') as fp:
            pickle.dump(dkappas,fp)
        with open('identities.pkl','wb') as fp:
            pickle.dump(identities,fp)
        with open('types.pkl','wb') as fp:
            pickle.dump(types,fp)
    else:
        with open('trajs.pkl','rb') as fp:
            trajs = pickle.load(fp)
        with open('grids.pkl','rb') as fp:
            grids = pickle.load(fp)
        with open('kappas.pkl','rb') as fp:
            dkappas = pickle.load(fp)
        with open('identities.pkl','rb') as fp:
            identities = pickle.load(fp)
        with open('types.pkl','rb') as fp:
            types = pickle.load(fp)

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
        types = { k:v for k,v in types.items() if k<index+1 }

    return grids, trajs, dkappas, identities, types

def node2gridinfo(grid, node, identity, increase_kappa):
    # For a node, return the required array
    if increase_kappa:
        node.kappas = node.kappas*grid.KAPPA_GROWTH_FACTOR
    return np.array([identity[0], identity[1],
            "*".join(['{:.8f}'.format(p) for p in node.loc]).encode(),
            "*".join(['{:.8e}'.format(kappa) for kappa in node.kappas]).encode(),
            node.type.encode()
        ],dtype=object)

def add_node_to_grid_info(grid,grid_info,node,increase_kappa=False,require_presence=False,verbose=False,finalize=False,reason=""):
    if isinstance(node,Node):
        # if it is a Node, we can use its identity to check whether it exists
        idx = np.where((np.array(grid_info[:,0],dtype=int)==node.identity[0]) & 
                       (np.array(grid_info[:,1],dtype=int)==node.identity[1]))[0]
        index = idx[0]
        identity = (grid_info[index][0],grid_info[index][1])
    else:
        # else use its location
        location_info = "*".join(['{:.8f}'.format(p) for p in node.loc]).encode()
        idx = np.where(grid_info[:,2] == location_info)[0]
        if idx.size == 0:
            identity = (grid_info[0][0],str(grid_info.shape[0]).encode())
        else:
            index = idx[0]
            identity = (grid_info[index][0],grid_info[index][1])

    if verbose: print(identity[0].decode(),identity[1].decode())

    if require_presence: 
        assert idx.size==1
        if not finalize:
            # Remove deviant trajectory
            #print("Remove {} trajs/traj_{}_{}.h5".format(reason,identity[0].decode(),identity[1].decode()))
            os.remove('trajs/traj_{}_{}.h5'.format(identity[0].decode(),identity[1].decode()))
        else:
            # This is to finalize an entry in grid file
            assert not increase_kappa
            new_entry = node2gridinfo(grid,node,identity,increase_kappa)
            grid_info[index][-1] = new_entry[-1] # change the type
            return grid_info

    if idx.size>0:
        assert idx.size==1
        # change the kappa values, the rest should be the same
        grid_info[index] = node2gridinfo(grid,node,identity,increase_kappa)
    else:
        grid_info = np.vstack((grid_info,node2gridinfo(grid,node,identity,increase_kappa)))

    return grid_info

def dump_grid(grid):
    from pathlib import Path
    Path('debug/').mkdir(parents=True, exist_ok=True)
    with open('debug/debug_grid_{}.txt'.format(grid.index),'w') as f:
        for n in grid.nodes:
            f.write("{}\t{}\n".format(",".join(['{: .8f}'.format(p) for p in n.loc]), n.type))

    if grid.major_grid is not None:
        with open('debug/debug_major_grid_{}.txt'.format(grid.index),'w') as f:
            for n in grid.major_grid.reference_nodes:
                f.write("{}\t{}\n".format(",".join(['{: .8f}'.format(p) for p in n.loc]), n.type))


def write_grid(grid,max_index=None,test=False):
    if test: # if test the user does not want any files to be created, changed, or removed
        return

    # First we check whether we need to write the following grid layer information
    if max_index is not None and grid.index+1 == max_index:
        print('MAX_LAYERS reached.')
    else:
        # If so, loop over all finer nodes and the non-refined reference nodes for these finer nodes
        next_grid_name = 'grid{0:0=2d}.txt'.format(grid.index+1)

        # if the file does not exist, just write the required nodes, if any finer nodes exist
        if len(grid.finer_nodes)>0:
            print('Checking next grid {}'.format(grid.index+1))
            if not os.path.exists(next_grid_name):
                with open(next_grid_name,'w') as f:
                    f.write('grid,nr,cvs,kappas,type\n')
                    idx = 0

                    for node in grid.finer_nodes:
                        f.write('{},{},{},{},{}\n'.format(grid.index+1,idx,"*".join(['{:.8f}'.format(p) for p in node.loc]),"*".join(['{:.8e}'.format(kappa) for kappa in node.kappas]),node.type))
                        idx += 1
                    for node in grid.reference_nodes:
                        if not node.refined:
                            f.write('{},{},{},{},{}\n'.format(grid.index+1,idx,"*".join(['{:.8f}'.format(p) for p in node.loc]),"*".join(['{:.8e}'.format(kappa) for kappa in node.kappas]),node.type))
                            idx += 1
            else:
                # Make the necessary replacements and additions
                # Read the grid file
                next_grid_information = np.genfromtxt(next_grid_name, delimiter=',',dtype=object,skip_header=1,encoding='utf')
                
                if next_grid_information.size==1: # problem with single line files
                    next_grid_information = np.array([next_grid_information])

                for node in grid.finer_nodes:
                    next_grid_information = add_node_to_grid_info(grid,next_grid_information,node,increase_kappa=False,require_presence=False)
                for node in grid.reference_nodes:
                    if not node.refined:
                        next_grid_information = add_node_to_grid_info(grid,next_grid_information,node,increase_kappa=False,require_presence=False)

                # Rewrite the grid_information file
                with open(next_grid_name,'w') as f:
                    f.write('grid,nr,cvs,kappas,type\n')
                    for gi in next_grid_information:
                        f.write(",".join([gin.decode() if isinstance(gin,bytes) else str(gin) for gin in list(gi)]) + '\n')

    # Second we check whether the current grid information should be updated
    # The corresponding grid file should exist per definition
    grid_name = 'grid{0:0=2d}.txt'.format(grid.index)
    assert os.path.exists(grid_name)

    # Read the grid file, everything will be read as a byte string
    grid_information = np.genfromtxt(grid_name, delimiter=',',dtype=object,skip_header=1,encoding='utf')
    if grid_information.size==1: # problem with single line files
        grid_information = np.array([grid_information])

    # Check all nodes
    #   grid.nodes contains non deviant nodes and refined reference nodes (which do not need to be considered below)
    #   if Node and deviant, replace with updated kappa taking max_kappa and growth factor into account
    #   if Reference node and not refined, replace with updated kappa ...
    print('Checking current grid {}'.format(grid.index))
    for node in grid.nodes:
        if isinstance(node,Node):
            if node.deviant:
                #print("Adding deviant node")
                grid_information = add_node_to_grid_info(grid,grid_information,node,increase_kappa=True,require_presence=True,reason='deviant')
            else:
                grid_information = add_node_to_grid_info(grid,grid_information,node,increase_kappa=False,require_presence=True,finalize=True)
            
        if isinstance(node,ReferenceNode):
            if not node.refined:
                if node.real:
                    #print("Adding refined reference node")
                    grid_information = add_node_to_grid_info(grid,grid_information,node,increase_kappa=True,require_presence=True,reason='unrefined reference')
                else:
                    #print('Realizing a virtual node')
                    grid_information = add_node_to_grid_info(grid,grid_information,node,increase_kappa=False,require_presence=False)
            else:
                if node.real:
                    # Finalize entry
                    grid_information = add_node_to_grid_info(grid,grid_information,node,increase_kappa=False,require_presence=True,finalize=True)

    # Rewrite the grid_information file
    with open(grid_name,'w') as f:
        f.write('grid,nr,cvs,kappas,type\n')
        for gi in grid_information:
            f.write(",".join([gin.decode() if isinstance(gin,bytes) else str(gin) for gin in list(gi)]) + '\n')


def plot_grid(grid):
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
        fig.set_size_inches(((grid.index+1)*4,1))
        ax.scatter(grid_grid[:,0],np.zeros_like(grid_grid[:,0]), s=0.2, color='k', label="Node")
        ax.scatter(np.array([node.loc[0] for node in grid.finer_nodes]),
     np.zeros_like(np.array([node.loc[0] for node in grid.finer_nodes])), s=0.05, color='r', label="Finer Node")
        ax.scatter(np.array([node.loc[0] for node in grid.deviant_nodes]),
     np.zeros_like(np.array([node.loc[0] for node in grid.deviant_nodes])), s=5., color='b',marker='x', label="Deviant Node")
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if not rnode.real and rnode.refined]),
     np.zeros_like(np.array([rnode.loc[0] for rnode in grid.reference_nodes if not rnode.real and rnode.refined])), s=0.05, color='g', marker='x', label="Virtual Reference Node")
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if not rnode.real and not rnode.refined]),
     np.zeros_like(np.array([rnode.loc[0] for rnode in grid.reference_nodes if not rnode.real and not rnode.refined])), s=0.05, color='orange', marker='x', alpha=0.5, label="Unrefined Virtual Reference Node")
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and not rnode.refined]),
     np.zeros_like(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and not rnode.refined])), s=0.05, color='orange', marker='x', alpha=0.5, label="Unrefined Realized Reference Node")
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and rnode.refined]),
     np.zeros_like(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and rnode.refined])), s=0.05, color='g', marker='x', alpha=0.5, label="Realized Reference Node")
    elif grid_grid.shape[1] == 2:
        fig.set_size_inches(((grid.index+1)*2,(grid.index+1)*2))
        ax.scatter(grid_grid[:,0],grid_grid[:,1], s=0.2, color='k')
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and rnode.refined]),
                   np.array([rnode.loc[1] for rnode in grid.reference_nodes if rnode.real and rnode.refined]),        s=0.2, color='g')
        ax.scatter(np.array([node.loc[0] for node in grid.finer_nodes]),
                   np.array([node.loc[1] for node in grid.finer_nodes]),                                              s=0.05, color='r', label="Finer Node")
        ax.scatter(np.array([node.loc[0] for node in grid.nodes if isinstance(node,Node) and node.deviant]),
                   np.array([node.loc[1] for node in grid.nodes if isinstance(node,Node) and node.deviant]),                                    s=2.0, linewidths=0.5, marker='s', facecolor='none', edgecolor='b', label="Deviant Node")
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if not rnode.real and rnode.refined]),
                   np.array([rnode.loc[1] for rnode in grid.reference_nodes if not rnode.real and rnode.refined]),    s=0.05, color='g', marker='x', label="Virtual Reference Node")
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if not rnode.real and not rnode.refined]),
                   np.array([rnode.loc[1] for rnode in grid.reference_nodes if not rnode.real and not rnode.refined]),s=0.05, color='r', marker='x', label="Unrefined Virtual Reference Node")
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and not rnode.refined]),
                   np.array([rnode.loc[1] for rnode in grid.reference_nodes if rnode.real and not rnode.refined]),    s=2.0, linewidths=0.5, marker='s', facecolor='none', edgecolor='orange', alpha=0.5, label="Unrefined Realized Reference Node")
        ax.scatter(np.array([rnode.loc[0] for rnode in grid.reference_nodes if rnode.real and rnode.refined]),
                   np.array([rnode.loc[1] for rnode in grid.reference_nodes if rnode.real and rnode.refined]),        s=2.0, linewidths=0.5, marker='s', facecolor='none', edgecolor='g', alpha=0.5, label="Realized Reference Node")
    else:
        pt.close('grid')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('grid_points_{}.pdf'.format(grid.index),bbox_inches='tight')
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
        grid.finer_nodes = [node for node in grid.finer_nodes if node.loc[i] <= edges['max'][i]+precision and node.loc[i] >= edges['min'][i]-precision]
        grid.reference_nodes = [rnode for rnode in grid.reference_nodes if rnode.loc[i] <= edges['max'][i]+precision and rnode.loc[i] >= edges['min'][i]-precision]


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
            
