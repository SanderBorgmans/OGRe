#! /usr/bin/python
import os,time,h5py,glob,pickle,sys,warnings,shutil
import numpy as np
import matplotlib.pyplot as pt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


from molmod.units import *
from ogre.post.nodes import *
from ogre.post.sampling_utils import precision

warnings.filterwarnings('ignore')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

########## LOAD FUNCTIONS ##########

def load_potential_file(file_loc):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module",file_loc)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_compressed_trajs(compressed_fn,number,locations,trajs,dkappas,identities,types):
    with h5py.File(compressed_fn,'r') as f:
        locations[number]   = np.array(f['locations'])
        trajs[number]   = np.array(f['trajs'])
        dkappas[number] = np.array(f['dkappas'])
        identities[number] = [tuple(identity) for identity in f['identities']]
        types[number] = [t for t in f['types']]

def load_trajs(fname,data,restart,locations,trajs,dkappas,identities,types):
    layer = np.genfromtxt(fname, delimiter=',',dtype=None,skip_header=1)

    if layer.size==1: # problem with single line files
        layer = np.array([layer])

    for point in layer:
        # Define identity
        lnr = int(point[0])
        nr = int(point[1])
        identity = (lnr,nr)
        dtype = point[4].decode()

        # Multi load: allows the loading of both compressed trajectories and non-compressed trajectories
        # This could occur in cases that the initial layer edges would be extended
        if 'multi_load' in data:
            multi_load = data['multi_load']
            # In this case some data points may have already been loaded
            if multi_load and identity in identities[lnr]:
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
            with h5py.File('trajs/traj_{}_{}.h5'.format(lnr,nr),'r') as f:
                tr = f['trajectory/cv_values'][:].reshape((1,-1,len(kappas)))

                if not lnr in locations.keys() or not lnr in trajs.keys():
                    locations[lnr]   = np.empty((0,len(kappas)))
                    trajs[lnr]   = np.empty((0,tr.shape[1],len(kappas)))
                    dkappas[lnr] = np.empty((0,len(kappas)))
                    identities[lnr] = []
                    types[lnr] = []


                locations[lnr]   = np.vstack((locations[lnr], cvs))
                trajs[lnr]   = np.vstack((trajs[lnr], tr))
                dkappas[lnr] = np.vstack((dkappas[lnr], kappas))
                identities[lnr].append(identity)
                types[lnr].append(dtype)

        except Exception:
            restart += '{},{},{},{},{}\n'.format(identity[0],identity[1],"*".join(['{:.8f}'.format(p) for p in cvs]),"*".join(['{:.8e}'.format(k) for k in kappas]),dtype)

    if len(restart)>0:
        gr = open('grid_restart.txt','w')
        gr.write('layer,nr,cvs,kappas,type\n')
        gr.write(restart)
        gr.close()
        print('Some simulations have to be restarted!')
        sys.exit(0)
        raise ValueError("Some simulations have to be restarted!")


def load_grid(data,index=None,verbose=True):
    """
        Create grid if it does not yet exist
    """
    # Load all layers and divide based on layernr
    locations = {}
    trajs = {}
    dkappas = {}
    identities = {}
    types = {}

    restart = ""

    multi_load = False
    if 'multi_load' in data:
        multi_load = data['multi_load']

    init = time.time()
    if not os.path.exists('trajs.pkl') or not os.path.exists('locations.pkl') or not os.path.exists('kappas.pkl'):
        layer_names = sorted(glob.glob('layer[0-9][0-9].txt'), key = lambda ln: int(ln.split('.')[-2].split('layer')[-1]))
        numbers = [int(ln.split('.')[-2].split('layer')[-1]) for ln in layer_names]
        if index is not None and max(numbers)<index:
            print('Max index reached: {} < {}'.format(max(numbers),index))
            sys.exit()

        for n,ln in enumerate(layer_names):
            number = numbers[n]

            # If we are extending the grid and already have some layers, this will not take subsequent layers into account
            if 'MAX_LAYERS' in data and not number < data['MAX_LAYERS']:
                continue

            compressed_fn = 'trajs/compressed_{}.h5'.format(number)
            compressed = os.path.exists(compressed_fn)
            if multi_load:
                # Load both the compressed part and new trajectories, this is convenient when expanding grid
                print("Loading both compressed file and trajectory files for layer {}.".format(number))
                assert compressed
                load_compressed_trajs(compressed_fn,number,locations,trajs,dkappas,identities,types)
                load_trajs(ln,data,restart,locations,trajs,dkappas,identities,types)
            else:
                if compressed:
                    print("Loading compressed file for layer {}.".format(number))
                    load_compressed_trajs(compressed_fn,number,locations,trajs,dkappas,identities,types)
                else:
                    print("Loading trajectory files for layer {}.".format(number))
                    load_trajs(ln,data,restart,locations,trajs,dkappas,identities,types)

        with open('trajs.pkl','wb') as fp:
            pickle.dump(trajs,fp)
        with open('locations.pkl','wb') as fp:
            pickle.dump(locations,fp)
        with open('kappas.pkl','wb') as fp:
            pickle.dump(dkappas,fp)
        with open('identities.pkl','wb') as fp:
            pickle.dump(identities,fp)
        with open('types.pkl','wb') as fp:
            pickle.dump(types,fp)
    else:
        with open('trajs.pkl','rb') as fp:
            trajs = pickle.load(fp)
        with open('locations.pkl','rb') as fp:
            locations = pickle.load(fp)
        with open('kappas.pkl','rb') as fp:
            dkappas = pickle.load(fp)
        with open('identities.pkl','rb') as fp:
            identities = pickle.load(fp)
        with open('types.pkl','rb') as fp:
            types = pickle.load(fp)

    if verbose:
        print("Loading trajectories took {} seconds.".format(time.time()-init))
    if index is not None:
        if index > max(locations.keys()):
            print('Max index reached: {} < {}'.format(max(locations.keys()),index))
            sys.exit(0)

        locations = { k:v for k,v in locations.items() if k<index+1 }
        trajs     = { k:v for k,v in trajs.items() if k<index+1 }
        dkappas   = { k:v for k,v in dkappas.items() if k<index+1 }
        identities = { k:v for k,v in identities.items() if k<index+1 }
        types     = { k:v for k,v in types.items() if k<index+1 }

    return locations, trajs, dkappas, identities, types

########## PLOT FUNCTIONS ##########

def scatter_nodes(ax,nodes,label,settings={}):
    if len(nodes)==0:
        return

    if len(nodes[0].loc)==1:
        ax.scatter(np.array([node.loc[0] for node in nodes]),
        np.zeros_like(np.array([node.loc[0] for node in nodes])), label=label, **settings)
    elif len(nodes[0].loc)==2:
        ax.scatter(np.array([node.loc[0] for node in nodes]),
                   np.array([node.loc[1] for node in nodes]), label=label, **settings)
    else:
        pass # no plot functionality for N>2 dimensions

def scatter(ax,nodes,n,size=1,labels=False,show_deviants=True):
    layer_nodes = np.array([node for node in nodes if isinstance(node,Node)])
    deviant_layer_nodes = np.array([node for node in nodes if not node.sane]) # this will capture all node and benchmarknodes that are deviant
    benchmark_superlayer_nodes = np.array([node for node in nodes if isinstance(node,SuperlayerBenchmarkNode)])
    benchmark_virtual_nodes = np.array([node for node in nodes if isinstance(node,VirtualBenchmarkNode)])
    benchmark_realized_nodes = np.array([node for node in nodes if isinstance(node,RealizedBenchmarkNode)])

    scatter_nodes(ax,layer_nodes,"Node" if labels else None,                                       settings={'s':0.30*size, 'c':colors[n]})
    if show_deviants:
        scatter_nodes(ax,deviant_layer_nodes,"Deviant" if labels else None,                        settings={'s':5.00*size, 'linewidths':0.5, 'marker':'s', 'facecolor':'none', 'edgecolor':'r'})
    scatter_nodes(ax,benchmark_superlayer_nodes,"Benchmark Node - superlayer" if labels else None, settings={'s':1.25*size, 'linewidths':0.5, 'marker':'s', 'facecolor':'none', 'edgecolor':colors[n-1]})
    scatter_nodes(ax,benchmark_virtual_nodes,"Benchmark Node - virtual" if labels else None,       settings={'s':0.10*size, 'c':colors[n-1], 'marker':'x'})
    scatter_nodes(ax,benchmark_realized_nodes,"Benchmark Node - realized" if labels else None,     settings={'s':1.25*size, 'c':colors[n-1], 'marker':'s'})

def plot_ref_grid(ax,grid,size=1.,show_deviants=True):
    tmp = grid.superlayer
    tmp_nodes = []
    while not tmp is None:
        tmp_nodes.append(tmp.nodes)
        tmp = tmp.superlayer
    for n,tmpn in enumerate(tmp_nodes[::-1]):
        scatter(ax,tmpn,n,size=size*0.25,show_deviants=show_deviants)
    scatter(ax,grid.nodes,len(tmp_nodes),size=size,labels=True,show_deviants=show_deviants) # add labels for the final scatter

def plot_dev(ax, node, traj, steps, layer):
    grid = layer.grid
    binwidths = grid.HISTOGRAM_BIN_WIDTHS
    bins = tuple([int(((grid.edges['max'][i]+grid.spacings[i])-(grid.edges['min'][i]-grid.spacings[i]))//binwidths[i]) for i in range(traj.shape[1])])

    h1, edges = np.histogramdd(traj[:], bins=bins, range=[(grid.edges['min'][i]-grid.spacings[i],
                                                           grid.edges['max'][i]+grid.spacings[i]) for i in range(traj.shape[1])], density=True)

    if len(node.loc) == 1:
        # (left corner, width, height) histogram starts at 0.1
        ax.add_patch(mpatches.Rectangle((node.loc[0]-steps[0], 0.1), 2*steps[0], np.max(h1)+0.25, facecolor='#e6e6e6', fill=True, zorder=-10))
        ax.bar(edges[0][:-1],h1,align='edge',width=binwidths,bottom=0.1,color=colors[layer.index+1],edgecolor='k',zorder=-1)
        ax.scatter([node.loc[0]],[0],s=20.0,marker='x',c=colors[1])
        ax.set_xlim([(node.loc[0]-3*steps[0]),(node.loc[0]+3*steps[0])])
        ax.set_ylim([-0.25,np.max(h1)+0.25])

        pt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the left edge are off
        right=False,       # ticks along the right edge are off
        labelleft=False)  # labels along the left edge are off

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
            
    elif len(node.loc) == 2:
        patch = mpatches.Rectangle((node.loc[0]-steps[0], node.loc[1]-steps[1]), 2*steps[0], 2*steps[1], facecolor='#e6e6e6', fill=True, zorder=-10) # (x,y), width, height
        h1m =  np.ma.masked_where(h1==0.0,h1)
        ax.pcolormesh(edges[0], edges[1], h1m.T, cmap='Oranges')


        #ax.scatter(traj[:,0],traj[:,1],s=0.1, c='r')
        ax.scatter([node.loc[0]],[node.loc[1]],s=20.0,marker='x',c='r')
        ax.add_patch(patch)
        ax.set_xlim([(node.loc[0]-4*steps[0]),(node.loc[0]+4*steps[0])])
        ax.set_ylim([(node.loc[1]-4*steps[1]),(node.loc[1]+4*steps[1])])
    else:
        raise NotImplementedError("Can't make deviation plots in more than 2 dimensions.")
    


def plot_consistency(ax, node, edges, hist_sample, hist_calc, binwidth, steps):
    if len(node.loc) == 1:
        # (left corner, width, height) histogram starts at 0.1
        ax.bar(edges[0][:-1],hist_sample,align='edge',width=binwidth,bottom=0.1,edgecolor='k',zorder=-1,label='sampled prob')
        ax.bar(edges[0][:-1],hist_calc,  align='edge',width=binwidth,bottom=0.1,label='biased prob', alpha=0.5)

        ax.scatter([node.loc[0]],[0],s=20.0,marker='x',c=colors[1])
        ax.set_ylim([-0.25,np.max(hist_sample)+0.25])

        pt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the left edge are off
        right=False,       # ticks along the right edge are off
        labelleft=False)  # labels along the left edge are off

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    elif len(node.loc) == 2:
        # for the consistency plot, we will plot the difference, using the limits of the histograms as scale
        diff = hist_sample-hist_calc
        diffm =  np.ma.masked_where(diff==0.0,diff)
        ax.pcolormesh(edges[0], edges[1], diffm.T, cmap='bwr',zorder=-2,vmin=-np.max(hist_sample),vmax=np.max(hist_sample))
        ax.scatter([node.loc[0]],[node.loc[1]],s=20.0,marker='x',c='r')
        ax.set_xlim([(node.loc[0]-4*steps[0]),(node.loc[0]+4*steps[0])])
        ax.set_ylim([(node.loc[1]-4*steps[1]),(node.loc[1]+4*steps[1])])
    else:
        raise NotImplementedError("Can't make consistency plots in more than 2 dimensions.")
    

    
def plot_overlap(ax, node1, node2, traj1, trajs2, steps, layer):
    grid = layer.grid
    binwidths = grid.HISTOGRAM_BIN_WIDTHS
    bins = tuple([int(((grid.edges['max'][i]+grid.spacings[i])-(grid.edges['min'][i]-grid.spacings[i]))//binwidths[i]) for i in range(traj1.shape[1])])

    h1, edges = np.histogramdd(traj1[grid.RUN_UP_TIME:], bins=bins, range=[(grid.edges['min'][i]-grid.spacings[i],
                                                                            grid.edges['max'][i]+grid.spacings[i]) for i in range(traj1.shape[1])], density=True)

    h2s = []
    for traj2 in trajs2:
        h2, edges = np.histogramdd(traj2[grid.RUN_UP_TIME:], bins=bins, range=[(grid.edges['min'][i]-grid.spacings[i],
                                                                                grid.edges['max'][i]+grid.spacings[i]) for i in range(traj2.shape[1])], density=True)
        h2s.append(h2)
    h2 = np.average(np.array(h2s),axis=0)


    if len(node1.loc) == 1:
        ax.scatter([node1.loc[0]],[0],s=30.0,marker='x',c=colors[1],label='node_1')
        ax.scatter([node2.loc[0]],[0],s=30.0,marker='x',c=colors[2],label='node_2')

        ax.bar(edges[0][:-1],h1,align='edge',width=binwidths,bottom=0.1,fill=False,edgecolor=colors[layer.index+1],zorder=-1)
        ax.bar(edges[0][:-1],h2,align='edge',width=binwidths,bottom=0.1,fill=False,edgecolor=colors[layer.index+2],zorder=-1)
        ax.bar(edges[0][:-1],np.minimum(h1,h2),align='edge',width=binwidths,bottom=0.1,color=colors[layer.index+3],zorder=0)#label='overlap')
        pt.xlim([(node1.loc[0]-4*steps[0]),(node1.loc[0]+4*steps[0])])

        pt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the left edge are off
        right=False,       # ticks along the right edge are off
        labelleft=False)  # labels along the left edge are off

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        
    elif len(node1.loc) == 2:
        ax.scatter([node1.loc[0]],[node1.loc[1]],s=30.0,marker='x',c='r',label='t1_loc',zorder=10)
        ax.scatter([node2.loc[0]],[node2.loc[1]],s=30.0,marker='x',c='b',label='t2_loc',zorder=10)

        if len(trajs2)>1:
            for t2 in trajs2:
                ax.scatter(t2[0,0],t2[0,1],s=40,marker='x', c='g')#, label='t2')

        h1m =  np.ma.masked_where(h1==0.0,h1)
        ax.pcolormesh(edges[0], edges[1], h1m.T, cmap='Reds',alpha=0.5,zorder=-2)

        h2m =  np.ma.masked_where(h2==0.0,h2)
        ax.pcolormesh(edges[0], edges[1], h2m.T, cmap='Blues',alpha=0.5,zorder=-2)

        hoverlap = np.minimum(h1,h2)
        h_overlapm = np.ma.masked_where(hoverlap==0.0,hoverlap)
        ax.pcolormesh(edges[0], edges[1], h_overlapm.T, cmap='Purples',zorder=-1)

        #ax.scatter(traj1[:,0],traj1[:,1],s=0.1, c='r')#, label='t1')
        #for t2 in trajs2:
        #    ax.scatter(t2[:,0],t2[:,1],s=0.1, c='b')#, label='t2')

        ax.set_xlim([(node1.loc[0]-4*steps[0]),(node1.loc[0]+4*steps[0])])
        ax.set_ylim([(node1.loc[1]-4*steps[1]),(node1.loc[1]+4*steps[1])])
    else:
        raise NotImplementedError("Can't make overlap plots in more than 2 dimensions.")

def plot_layer(layer):
    fig = pt.figure('layer')
    ax = fig.gca()
    # Plot previous layers
    tmp = layer
    while not tmp.superlayer is None:
        tmp = tmp.superlayer
        tmp_layer = np.array([node.loc for node in tmp.nodes])
        if tmp_layer.shape[1] == 1:
            ax.scatter(tmp_layer[:,0], np.zeros_like(tmp_layer[:,0]), s=0.1, color='gray')
            ax.scatter(np.array([node.loc for node in tmp.nodes if isinstance(node,Node) and node.extreme_kappa]),
         np.zeros_like(np.array([node.loc for node in tmp.nodes if isinstance(node,Node) and node.extreme_kappa])), s=0.1, marker='x', color='gray')
        elif tmp_layer.shape[1] == 2:
            ax.scatter(tmp_layer[:,0],tmp_layer[:,1], s=0.1, color='gray')
            ax.scatter(np.array([node.loc[0] for node in tmp.nodes if isinstance(node,Node) and node.extreme_kappa]),
                       np.array([node.loc[1] for node in tmp.nodes if isinstance(node,Node) and node.extreme_kappa]), s=0.1, marker='x', color='gray')
        else:
            break

    # Plot current grid and refinement
    layer_loc = np.array([node.loc for node in layer.nodes])
    if layer_loc.shape[1] == 1:
        fig.set_size_inches(((layer.index+1)*4,1))
        pt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,        # ticks along the left edge are off
        right=False,       # ticks along the right edge are off
        labelleft=False)  # labels along the left edge are off
    elif layer_loc.shape[1] == 2:
        fig.set_size_inches(((layer.index+1)*2,(layer.index+1)*2))
    else:
        pt.close('layer')
        return
    
    # Plot current nodes
    scatter_nodes(ax,layer.nodes,label="Node",settings={'s':0.2, 'color':'k'})


    # Highlight deviant nodes
    scatter_nodes(ax,[node for node in layer.nodes if not node.sane],
                  label="Deviant - Node",
                  settings={'s':2.0, 'linewidths':0.5, 'marker':'s', 'facecolor':'none', 'edgecolor':'blue', 'alpha':0.5})

    # Plot sublayer nodes
    scatter_nodes(ax,[node for node in layer.sublayer_nodes if isinstance(node,Node)],
                  label="Sublayer Node",
                  settings={'s':0.05, 'color':'r'})

    # These scatters place hues on top of nodes to indicate their relevance with respect to the sublayer nodes
    # Plot corresponding sane superlayer benchmark nodes
    scatter_nodes(ax,[node for node in layer.sublayer_nodes if isinstance(node,SuperlayerBenchmarkNode) and node.sane],
                  label="Benchmark Node - superlayer",
                  settings={'s':2.0, 'linewidths':0.5, 'marker':'s', 'facecolor':'none', 'edgecolor':'g', 'alpha':0.5})

    # Plot corresponding deviant superlayer benchmark nodes - which should be replaced by realized benchmark nodes - this should never occur
    scatter_nodes(ax,[node for node in layer.sublayer_nodes if isinstance(node,SuperlayerBenchmarkNode) and not node.sane],
                  label="Deviant - Benchmark Node - superlayer",
                  settings={'s':2.0, 'linewidths':0.5, 'marker':'s', 'facecolor':'none', 'edgecolor':'r', 'alpha':0.5})

    # Plot corresponding virtual benchmark nodes 
    scatter_nodes(ax,[node for node in layer.sublayer_nodes if isinstance(node,VirtualBenchmarkNode) and node.sane],
                  label="Benchmark Node - virtual",
                  settings={'s':0.05, 'color':'g', 'marker':'x'})

    # Plot corresponding deviant virtual benchmark nodes - which should be replaced by realized benchmark nodes - this should never occur
    scatter_nodes(ax,[node for node in layer.sublayer_nodes if isinstance(node,VirtualBenchmarkNode) and not node.sane],
                  label="Deviant - Benchmark Node - virtual",
                  settings={'s':0.05, 'color':'r', 'marker':'x'})

    # Plot corresponding realized benchmark nodes 
    scatter_nodes(ax,[node for node in layer.sublayer_nodes if isinstance(node,RealizedBenchmarkNode)],
                  label="Benchmark Node - realized",
                  settings={'s':0.075, 'color':'g'})

    # Plot corresponding deviant realized benchmark nodes - this will overlay the existing benchmark nodes
    scatter_nodes(ax,[node for node in layer.sublayer_nodes if isinstance(node,RealizedBenchmarkNode) and not node.sane],
                  label="Deviant - Benchmark Node - realized",
                  settings={'s':2.0, 'linewidths':0.5, 'marker':'s', 'facecolor':'none', 'edgecolor':'r', 'alpha':0.5})


    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),frameon=False)
    fig.savefig('grid_points_{}.pdf'.format(layer.index),bbox_inches='tight')
    pt.close('layer')


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


def dump_layer(layer):
    from pathlib import Path
    Path('debug/').mkdir(parents=True, exist_ok=True)
    with open('debug/debug_layer_{}.txt'.format(layer.index),'w') as f:
        for n in layer.nodes:
            f.write("{}\t{}\n".format(",".join(['{: .8f}'.format(p) for p in n.loc]), n.type))

    with open('debug/debug_sublayer_{}.txt'.format(layer.index),'w') as f:
        for n in layer.sublayer_nodes:
            f.write("{}\t{}\n".format(",".join(['{: .8f}'.format(p) for p in n.loc]), n.type))

    if layer.superlayer is not None:
        with open('debug/debug_sublayer_superlayer_{}.txt'.format(layer.index),'w') as f:
            for n in layer.superlayer.sublayer_nodes:
                f.write("{}\t{}\n".format(",".join(['{: .8f}'.format(p) for p in n.loc]), n.type))


def add_node_to_grid_info(layer,layer_info,node,require_presence=False,verbose=False):
    if node.identity is not None:
        idx = np.where((np.array(layer_info[:,0],dtype=int)==node.identity[0]) & 
                       (np.array(layer_info[:,1],dtype=int)==node.identity[1]))[0]
        index = idx[0]
        identity = parse_identity((layer_info[index][0],layer_info[index][1]))
    else:
        # else use its location
        location_info = "*".join(['{:.8f}'.format(p) for p in node.loc]).encode()
        idx = np.where(layer_info[:,2] == location_info)[0]
        if idx.size == 0:
            identity = (layer_info[0][0],layer_info.shape[0])
        else:
            index = idx[0]
            identity = (layer_info[index][0],layer_info[index][1])
        identity = parse_identity(identity)

    if verbose: print(identity[0],identity[1])

    if require_presence: 
        if not node.sane:
            # Remove deviant trajectory
            #print("Remove {} trajs/traj_{}_{}.h5".format(reason,identity[0],identity[1]))
            os.remove('trajs/traj_{}_{}.h5'.format(identity[0],identity[1]))

    if idx.size>0:
        assert idx.size==1
        layer_info[index] = node.node_info(grid=layer.grid,identity=identity)
    else:
        layer_info = np.vstack((layer_info,node.node_info(grid=layer.grid,identity=identity)))

    return layer_info



def write_layer(layer,max_index=None,debug=False):
    if debug: # if debug the user does not want any files to be created, changed, or removed
        return

    # First we check whether we need to write the following grid layer information
    if max_index is not None and layer.index+1 == max_index:
        # Reset the non_converged file
        if os.path.exists('non_converged'):
            os.remove('non_converged')
        if len(layer.sublayer_nodes)>0:
            with open('non_converged','w') as f:
                pass # the file does not need any input
        print('MAX_LAYERS reached.')
    else:
        # If so, loop over all sublayer nodes (nodes and benchmark nodes)
        next_layer_name = 'layer{0:0=2d}.txt'.format(layer.index+1)

        # if the file does not exist, just write the required nodes, if any sublayer nodes exist
        if len(layer.sublayer_nodes)>0:
            print('Checking next layer {}'.format(layer.index+1))
            if not os.path.exists(next_layer_name):
                next_layer_information = np.empty((0,5))
                idx = 0

                # For a new file, only new nodes should be present
                for node in layer.sublayer_nodes:
                    node_info = None
                    if isinstance(node,Node):
                        assert isinstance(node,NewNode)
                        node_info = node.node_info(identity=(layer.index+1,idx),grid=layer.grid)
                    elif isinstance(node,RealizedBenchmarkNode):
                        assert isinstance(node.node, NewNode)
                        node_info = node.node_info(identity=(layer.index+1,idx),grid=layer.grid)
                    else:
                        continue

                    next_layer_information = np.vstack((next_layer_information,node_info))
                    idx+=1

                with open(next_layer_name,'w') as f:
                    f.write('layer,nr,cvs,kappas,type\n')
                    for li in next_layer_information:
                        f.write(",".join([lin.decode() if isinstance(lin,bytes) else str(lin) for lin in list(li)]) + '\n')

            else:
                # Make the necessary replacements and additions
                # Read the layer information
                next_layer_information = np.genfromtxt(next_layer_name, delimiter=',',dtype=object,skip_header=1,encoding='utf')

                if len(next_layer_information.shape)==1: # problem with single line files
                    next_layer_information = np.array([next_layer_information])

                for node in layer.sublayer_nodes:
                    # Only realized or to be realized nodes should be considered to be written to the simulation file
                    if isinstance(node, Node): # this contains node, new_node, and deviant_node
                        next_layer_information = add_node_to_grid_info(layer,next_layer_information,node,require_presence=False)
                    elif isinstance(node,RealizedBenchmarkNode):
                        next_layer_information = add_node_to_grid_info(layer,next_layer_information,node,require_presence=False)
                    else:
                        continue

                # Rewrite the grid_information file
                with open(next_layer_name,'w') as f:
                    f.write('layer,nr,cvs,kappas,type\n')
                    for li in next_layer_information:
                        f.write(",".join([lin.decode() if isinstance(lin,bytes) else str(lin) for lin in list(li)]) + '\n')

    # Second we check whether the current layer information should be updated
    # The corresponding layer file should exist per definition
    layer_name = 'layer{0:0=2d}.txt'.format(layer.index)
    assert os.path.exists(layer_name)

    # Read the layer file, everything will be read as a byte string
    layer_information = np.genfromtxt(layer_name, delimiter=',',dtype=object,skip_header=1,encoding='utf')
    if len(layer_information.shape)==1: # problem with single line files
        layer_information = np.array([layer_information])

    # Check all nodes, layer.nodes contains:
    #   sane nodes and sane benchmark nodes (which do not need to be considered below)
    #   if Node and not sane, the kappa is automatically increased when parsing node_info, taking max_kappa and growth factor into account
    #   if Superlayer Benchmark Node (has to be sane per definition, no deviant superlayer benchmark can exist) 
    #   if Virtual Benchmark Node and not sane, replace with RealizedBenchmarkNode
    #   if Realized Benchmark Node node and not sane, the kappa is automatically increased when parsing node_info, taking max_kappa and growth factor into account
    print('Checking current grid {}'.format(layer.index))
    for node in layer.nodes:
        if isinstance(node,Node):
            layer_information = add_node_to_grid_info(layer,layer_information,node,require_presence=True)
        elif isinstance(node,RealizedBenchmarkNode):
            layer_information = add_node_to_grid_info(layer,layer_information,node,require_presence=(not isinstance(node.node,NewNode)))
        elif (isinstance(node,VirtualBenchmarkNode) and not node.sane):
            raise ValueError('This CANNOT happen!')
            #layer_information = add_node_to_grid_info(layer,layer_information,node,require_presence=False)
        elif (isinstance(node,SuperlayerBenchmarkNode) and not node.sane):
            raise ValueError('This CANNOT happen!')
        else:
            continue

    # Rewrite the grid_information file
    with open(layer_name,'w') as f:
        f.write('layer,nr,cvs,kappas,type\n')
        for li in layer_information:
            f.write(",".join([lin.decode() if isinstance(lin,bytes) else str(lin) for lin in list(li)]) + '\n')

#############################

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

def write_colvars(filename,rtrajs,rlocations,rkappas,verbose=True):
    if not os.path.exists('colvars/'):
        os.mkdir('colvars')

    with open(filename,'w') as g:
        for n,traj in enumerate(rtrajs):
            # Define point
            cvs = rlocations[n]
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
            
def remove_colvars(filename):
    if os.path.exists('colvars/'):
        shutil.rmtree('colvars')

    if os.path.exists(filename):
            os.remove(filename)


def format_layer_files(data,debug=False,verbose=True):
    if debug: # if debug the user is only interested in the possible outcome
        return
    # Get all compressed trajectory identities
    identities_compressed = []
    ctnames = glob.glob('trajs/compressed_*.h5')
    for ct in ctnames:
        with h5py.File(ct,'r') as hct:
            identities_ct = [tuple(val) for val in hct['identities']]
            for ict in identities_ct:
                identities_compressed.append(ict)

    # Get all trajectory identities
    tnames = glob.glob('trajs/traj_*_*.h5')
    ids = [tn.split('/')[1].split('.')[0].split('_') for tn in tnames]
    identities_traj = [(int(i[1]),int(i[2])) for i in ids]

    # Get all gridpoint identities
    lnames = glob.glob('layer*.txt')
    identities_lp = []
    lp_lines = {}
    for ln in lnames:
        lnr = int(ln.split('.')[0][5:])
        if 'MAX_LAYERS' in data and not lnr < data['MAX_LAYERS']:
            continue
        with open(ln,'r') as f:
            lines = f.readlines()
        lines = lines[1:] # skipheader
        for l in lines:
            id_line = l.split(',')[:2]
            id = (int(id_line[0]),int(id_line[1]))
            data = l.split(',')[2:]
            identities_lp.append(id)
            lp_lines[id] = data

    # Find those elements in identities_gp that are not in identities_traj and sort them for clean grid file
    identities = list(set(identities_lp)-set(identities_traj)-set(identities_compressed))
    identities = sorted(identities, key=lambda tup: (tup[0],tup[1]))

    with open('run.txt','w') as f:
        f.write('layer,nr,cvs,kappas,type\n')
        for id in identities:
            f.write('{},{},{}'.format(id[0],id[1],",".join(lp_lines[id])))

    if len(identities)==0 and verbose:
        print('It might be a good idea to compress your data using ogre_compress_iteration.py, if you do not need the trajectory data for other purposes.')
