#! /usr/bin/python
import numpy as np
from scipy.spatial import KDTree
from molmod.units import *
from ogre.post.sampling_utils import *
from ogre.input.utils import get_cv_units


# NOTE #
# The grid is defined in terms of 'cv_units' to avoid conversion errors
# Some quantities/constants are defined in sampling_utils.py

def scatter(ax,grid,n,size=1,labels=False):
    real_grid = np.array([node.loc for node in grid if isinstance(node,Node)])
    reference_grid_real = np.array([node.loc for node in grid if isinstance(node,ReferenceNode) and node.real and not node.refined])
    reference_grid_real_refined = np.array([node.loc for node in grid if isinstance(node,ReferenceNode) and node.real and node.refined])
    reference_grid_virtual = np.array([node.loc for node in grid if isinstance(node,ReferenceNode) and not node.real])
    if real_grid.shape[1] == 1:
        if real_grid.size > 0: ax.scatter(real_grid[:,0],np.zeros_like(real_grid[:,0]), s=0.3*size, c=colors[n], label="Node" if labels else None)
        if reference_grid_real.size > 0: ax.scatter(reference_grid_real[:,0],np.zeros_like(reference_grid_real[:,0]), s=0.75*size, c=colors[n-1], marker='s', label="Unrefined Reference Node" if labels else None)
        if reference_grid_real_refined.size > 0: ax.scatter(reference_grid_real_refined[:,0],np.zeros_like(reference_grid_real_refined[:,0]), s=0.75*size, c=colors[n-1], marker='D', label="Realized Reference Node" if labels else None)
        if reference_grid_virtual.size > 0: ax.scatter(reference_grid_virtual[:,0],np.zeros_like(reference_grid_virtual[:,0]), s=0.7*size, c=colors[n-1], marker='+', label="Virtual Reference Node" if labels else None)
    elif real_grid.shape[1] == 2:
        if real_grid.size > 0: ax.scatter(real_grid[:,0],real_grid[:,1], s=0.3*size, c=colors[n], label="Node" if labels else None)
        if reference_grid_real.size > 0: ax.scatter(reference_grid_real[:,0],reference_grid_real[:,1], s=0.75*size, c=colors[n-1], marker='s', label="Unrefined Reference Node" if labels else None)
        if reference_grid_real_refined.size > 0: ax.scatter(reference_grid_real_refined[:,0],reference_grid_real_refined[:,1], s=1.25*size, c=colors[n-1], marker='D', label="Realized Reference Node" if labels else None)
        if reference_grid_virtual.size > 0: ax.scatter(reference_grid_virtual[:,0],reference_grid_virtual[:,1], s=0.7*size, c=colors[n-1], marker='+', label="Virtual Reference Node" if labels else None)


def plot_ref_grid(ax,grid,size=1.):
    tmp = grid.major_grid
    tmp_nodes = []
    while not tmp is None:
        tmp_nodes.append(tmp.nodes)
        tmp = tmp.major_grid
    for n,tmpn in enumerate(tmp_nodes[::-1]):
        scatter(ax,tmpn,n,size=size*0.25)
    scatter(ax,grid.nodes,len(tmp_nodes),size=size,labels=True) # add labels for the final scatter


def get_node_from_nodes(tree,point):
    found,index = find_node_index(tree,point)
    if found: return index

def create_grid(index,grid,trajs,kappas,identities,types,data,major_grid=None):
    """
        This function returns a Grid object based on the grid points and the trajectories it receives
        The grid points should be defined on a single distance scale (for each refinement a different grid should be defined)
        * Arguments
            grid            NxD array with all grid points
            trajs           NxMxD array with trajectory of M steps for every grid point
            kappas          NxD array with all kappa values
            identities      list with N entries, with the identity of each grid point
            types           list with N entries, the type of that grid point
            data            The yaml data file
            major_grid      The previous grid object, containing all previous information
    """
    # Get units
    cv_units = get_cv_units(data)

    # Define variables
    nodes = []
    edges = data['edges']
    spacings = data['spacings']

    # Scale quantities that are not defined in terms of units
    trajs /= cv_units

    steps = np.array(spacings)/(2**index)
    print(steps)
    if any(steps/2.<precision): raise ValueError('Making another refinement would require increasing DECIMALS. Adapt this in the sampling_utils.py file.')

    # Create a KDTree with the reference nodes, to update them
    ref_tree = None
    if major_grid is not None:
        ref_tree = KDTree(np.asarray([node.loc for node in major_grid.reference_nodes]))

    for n,point in enumerate(grid):
        if types[n] in ['node','new_node','deviant_node']:
            # the sanity of deviant nodes will be checked later
            nodes.append(Node(point,trajs[n],kappas[n],identities[n]))
        elif types[n] in ['reference_node','unrefined_reference_node','unrefined_virtual_reference_node']: 
            # this requires a major grid, and it should exist by definition if reference nodes are present
            assert major_grid is not None
            
            # identify the simulation with a node from the major_grid reference_nodes and replace it independent of the type
            rn_idx = get_node_from_nodes(ref_tree,point)

            # assume everything is fine
            rn_tmp = ReferenceNode(point,[Node(point,trajs[n],kappas[n],identities[n])],real=True,refined=True)

            if types[n] == 'reference_node':
                # this is a refined reference node which can just be added
                pass
            elif types[n] in ['unrefined_reference_node','unrefined_virtual_reference_node']:
                # check whether the simulation is now confined
                confinement = major_grid.traj_confinement(rn_tmp,data,factor=2.) # by choosing a factor 2 we are effectively checking the confinement for the current grid
                rn_tmp.refined = confinement>=major_grid.CONFINEMENT_THR # set the refined state
            else:
                raise ValueError('How did I get here?')
            
            major_grid.reference_nodes[rn_idx] = rn_tmp

    # Add the reference grid points to the grid
    if not major_grid is None:
        for node in major_grid.reference_nodes:
            nodes.append(node)

    # Assign constants from data or assing default values
    RUN_UP_TIME         = data['runup']
    CONFINEMENT_THR     = data['CONFINEMENT_THR']     if 'CONFINEMENT_THR'     in data else 0.30
    OVERLAP_THR         = data['OVERLAP_THR']         if 'OVERLAP_THR'         in data else 0.30
    KAPPA_GROWTH_FACTOR = data['KAPPA_GROWTH_FACTOR'] if 'KAPPA_GROWTH_FACTOR' in data else 2.
    MAX_KAPPA           = data['MAX_KAPPA']           if 'MAX_KAPPA'           in data else None

    return Grid(index,major_grid,nodes,edges,steps,RUN_UP_TIME,CONFINEMENT_THR,OVERLAP_THR,KAPPA_GROWTH_FACTOR,MAX_KAPPA=MAX_KAPPA)


class Node(object):
    def __init__(self, loc, traj, kappas, identity):
        self.loc = loc
        self.trajs = [traj]
        self.kappas = kappas
        self.identity = identity
        self.neighbours = []
        self.extreme_kappa = False
        self.deviant = False # True corresponds to not refined
        self._type = None

    def set_extreme_kappa(self,MAX_KAPPA,KAPPA_GROWTH_FACTOR):
        if np.any(MAX_KAPPA<self.kappas*KAPPA_GROWTH_FACTOR):
            self.extreme_kappa = True

    @property
    def type(self):
        self._type = 'node' if not self.deviant else 'deviant_node'
        return self._type


class FinerNode(object):
    def __init__(self, loc, kappas, MAX_KAPPA=None, identity=None):
        self.loc = loc
        self.kappas = np.clip(kappas, 0, MAX_KAPPA if MAX_KAPPA is not None else np.inf)
        self.identity = identity
        self._type = None

    @property
    def type(self):
        self._type = 'new_node'
        return self._type
        

class ReferenceNode(object):
    def __init__(self, loc, nodes, real=False, refined=False):
        self.loc = loc
        self.nodes = nodes
        self.trajs = [traj for node in self.nodes for traj in node.trajs]
        self.kappas = np.max([node.kappas for node in self.nodes],axis=0)
        self.neighbours = []
        self.real = real
        self.refined = refined # either True or False, True when the Reference Node is ready to be compared with
        # when a virtual Reference Node needs to be refined it will be realized first, after which it will then be refined
        # as we assume that that Reference node can no longer be represented by surrounding nodes and a trajectory has to be calculated for this point in the next iteration
        self._type = None

    def realize(self,kappas):
        assert not self.real
        self.kappas = kappas
        self.refined = False

    def refine(self,KAPPA_GROWTH_FACTOR,MAX_KAPPA):
        self.kappas =  np.clip(self.kappas * KAPPA_GROWTH_FACTOR, 0, MAX_KAPPA if MAX_KAPPA is not None else np.inf)
        self.refined = False

    @property
    def type(self):
        if self.real:
            if self.refined: self._type = 'reference_node'
            else: self._type = 'unrefined_reference_node'
        else:
            if self.refined: self._type = 'virtual reference_node'
            else: self._type = 'unrefined_virtual_reference_node'  
        return self._type


class Grid(object):
    def __init__(self, index, major_grid, nodes, edges, steps, RUN_UP_TIME, CONFINEMENT_THR, OVERLAP_THR, KAPPA_GROWTH_FACTOR,MAX_KAPPA=None):
        self.index = index
        self.major_grid = major_grid
        self.nodes = nodes # these are the nodes/reference nodes that contain trajectories
        self.edges = edges
        self.steps = steps
        self.finer_nodes = [] # this will be calculated
        self.reference_nodes = [] # this will be calculated
        self.deviant_nodes = [] # this will be calculated
        self.realized_virtual_reference_nodes = [] # this will be calculated
        self.overlap = None
        # CONSTANTS
        self.RUN_UP_TIME = RUN_UP_TIME
        self.CONFINEMENT_THR = CONFINEMENT_THR
        self.OVERLAP_THR = OVERLAP_THR
        self.KAPPA_GROWTH_FACTOR = KAPPA_GROWTH_FACTOR
        self.MAX_KAPPA = MAX_KAPPA

        # Identify extreme kappas, these nodes will be completely skipped during comparison
        # We assume that these nodes are not interesting
        self.identify_extreme_kappas()


    def identify_extreme_kappas(self):
        if not self.MAX_KAPPA is None:
            for node in self.nodes:
                if isinstance(node, Node):
                    node.set_extreme_kappa(self.MAX_KAPPA, self.KAPPA_GROWTH_FACTOR)


    def characterize_nodes(self):
        """
            Find and assign the neighbours for each point
        """
        neighbours = find_neighbours(self.steps,self.nodes)
        for n,node in enumerate(self.nodes):
            node.neighbours = neighbours[n]


    def traj_confinement(self,node,data=None,factor=1.):
        trajs = node.trajs

        t = trajs[0][self.RUN_UP_TIME:]
        fullt = trajs[0]
        if len(trajs)>1:
            for traj in trajs[1:]:
                t = np.vstack((t,traj[self.RUN_UP_TIME:]))
                fullt = np.vstack((t,traj))

        mask = np.ones(t.shape[0],dtype=bool)
        for n,step in enumerate(self.steps):
            mask = mask & (t[:,n]<node.loc[n]+step/factor) & (t[:,n]>node.loc[n]-step/factor)
        confinement = sum(mask)/float(len(mask))

        if not data is None and 'plot_con' in data and data['plot_con'] and confinement < self.CONFINEMENT_THR:
            fig = pt.figure()
            ax = fig.gca()
            plot_ref_grid(ax,self,size=50.0)
            plot_dev(ax, node, fullt, self.steps)
            fig.suptitle("Confinement = {:4.3f}".format(confinement))
            pt.show()
        return confinement


    def NNoverlap(self,node1,node2,data=None):
        trajs1 = node1.trajs
        trajs2 = node2.trajs

        overlap = 0
        assert len(trajs1)==1 # this has to be for non reference node
        t1 = trajs1[0]

        if data is not None:
            binwidths = data['HISTOGRAM_BIN_WIDTHS']
            bins = tuple([int(((self.edges['max'][i]+data['spacings'][i])-(self.edges['min'][i]-data['spacings'][i]))//binwidths[i]) for i in range(t1.shape[1])])
        else:
            bins = 201
        
        #try:
        h1, edges = np.histogramdd(t1[self.RUN_UP_TIME:], bins=bins, range=[(self.edges['min'][i]-data['spacings'][i],
                                                                                self.edges['max'][i]+data['spacings'][i]) for i in range(t1.shape[1])], density=True)
        #except FloatingPointError:
            # This happens if the trajectory falls outside of the edges


        bin_volume = np.prod([np.abs(edge[1] - edge[0]) for edge in edges])
        for t2 in trajs2:
            # Convert traj to histogram
            h2, _ = np.histogramdd(t2[self.RUN_UP_TIME:], bins=bins, range=[(self.edges['min'][i]-data['spacings'][i],
                                                                            self.edges['max'][i]+data['spacings'][i]) for i in range(t1.shape[1])], density=True)
            # Check overlap of histograms
            overlap += np.sum(np.minimum(h1,h2)*bin_volume)
        overlap /= len(trajs2)

        if not data is None and 'plot_overlap' in data and data['plot_overlap'] and overlap < self.OVERLAP_THR:
            fig = pt.figure()
            ax = fig.gca()
            plot_ref_grid(ax,self,size=10.0)
            plot_overlap(ax, node1, node2, t1, trajs2, self.steps)
            fig.legend()
            fig.suptitle("Overlap = {}".format(overlap))
            pt.show()
        return overlap


    def check_overlap(self, index, data):
        """
            Check overlap for grid.
            The plot functionality only works for ND grids with N < 3
        """
        # Assign al nearest neighbours
        self.characterize_nodes()

        self.overlap = np.ones((len(self.nodes),2*len(self.steps)), dtype=bool) # default everything overlaps

        if len(data['spacings']) == 1:
            fig = pt.figure('check_overlap',figsize=((index+1)*4,2))
        elif len(data['spacings']) == 2:
            fig = pt.figure('check_overlap',figsize=((index+1)*2,(index+1)*2))
        else:
            fig = pt.figure('check_overlap')
            print("Plotting the overlap data is not implemented for more than 2 dimension!")
        ax = fig.gca()

        # Plot all previous grids as a reference for this grid
        plot_ref_grid(ax,self,size=1.)
        pt.close('check_overlap') # make sure this does not get plotted somewhere

        # Check if certain simulations deviatiated substantially from their supposed locations
        for n,node in enumerate(self.nodes):
            if isinstance(node, Node) and not node.extreme_kappa:
                confinement = self.traj_confinement(node,data)
                # If the simulation went out of bounds, the overlap loses all meaning at this location
                if confinement < self.CONFINEMENT_THR:
                    ##self.deviant[n] = True
                    node.deviant = True
                    if len(node.loc)==1:
                        ax.scatter(node.loc[0],0., s=5., color='b', marker='x')
                    elif len(node.loc)==2:
                        ax.scatter(node.loc[0],node.loc[1], s=5., color='b', marker='x')

        # Check overlap between point and neighbours in idx to find points for which finer grid is required
        for n,node in enumerate(self.nodes):
            if isinstance(node, Node) and not node.extreme_kappa:
                # Check overlap for all Nodes (not reference nodes)
                # Plot overlap grid to see whether nearest neighbour assignment works
                for m,neighbour in enumerate(node.neighbours):
                     if len(node.loc)==1:
                         ax.plot((node.loc[0],self.nodes[neighbour].loc[0]), (0,0) , linestyle=':' ,color='gray', lw=0.3)
                     elif len(node.loc)==2:
                         ax.plot((node.loc[0],self.nodes[neighbour].loc[0]),
                                 (node.loc[1],self.nodes[neighbour].loc[1]), linestyle=':' ,color='gray', lw=0.3)
                 # Only consider overlap if this node is not deviant (else overlap loses meaning)
                if not node.deviant:
                    self.overlap[n,:len(node.neighbours)] = False # initialize all neighbouring overlap values to be faulty
                    for m,neighbour in enumerate(node.neighbours):
                        # Make a distinction between real and virtual ReferenceNodes and other Nodes
                        # - If the node is a virtual reference node and the overlap is faulty we should perform a simulation at that location instead
                        # - If the node is a real reference node we should also check whether the Node it was based was not deviant (it might still be refined)
                        # - If the node is another node we should check whether it is deviant before comparing
                        if isinstance(self.nodes[neighbour],ReferenceNode) and self.nodes[neighbour].refined and not self.nodes[neighbour].real:
                            overlap = self.NNoverlap(node, self.nodes[neighbour], data)>=self.OVERLAP_THR
                            if not overlap:
                                # As we assume virtual Reference Nodes are initially sane, a faulty overlap check will result in a realization and later refinement
                                self.nodes[neighbour].realize(node.kappas)
                                if len(node.loc)==1:
                                    ax.plot((node.loc[0],self.nodes[neighbour].loc[0]), (0,0) , color='orange', lw=0.3)
                                elif len(node.loc)==2:
                                    ax.plot((node.loc[0],self.nodes[neighbour].loc[0]),
                                            (node.loc[1],self.nodes[neighbour].loc[1]) , color='orange', lw=0.3)
                            self.overlap[n,m] = True

                        elif isinstance(self.nodes[neighbour],ReferenceNode) and self.nodes[neighbour].refined and self.nodes[neighbour].real:
                            self.overlap[n,m] = self.NNoverlap(node, self.nodes[neighbour], data)>=self.OVERLAP_THR
                        
                        elif isinstance(self.nodes[neighbour],Node) and not self.nodes[neighbour].deviant:
                            self.overlap[n,m] = self.NNoverlap(node, self.nodes[neighbour], data)>=self.OVERLAP_THR
                        else:
                            self.overlap[n,m] = True

                        if not self.overlap[n,m]:
                            if len(node.loc)==1:
                                ax.plot((node.loc[0],self.nodes[neighbour].loc[0]), (0,0) , color='r', lw=0.3)
                            elif len(node.loc)==2:
                                ax.plot((node.loc[0],self.nodes[neighbour].loc[0]),
                                        (node.loc[1],self.nodes[neighbour].loc[1]), color='r', lw=0.3)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig('overlap_{}.pdf'.format(index),bbox_inches='tight')

    def refine_grid_nodes(self):
        # Create a grid of finer and reference nodes for overlap refinement
        # Create set of unique node pairs where overlap is faulty
        overlap_pairs = np.array(np.where(self.overlap==0)).T  # with transpose the shape is (N,2)
        # Convert node and neighbour number to node numbers
        overlap_pairs = list(set([tuple(sorted([pair[0],self.nodes[pair[0]].neighbours[pair[1]]])) for pair in overlap_pairs]))

        dims = (2*(len(self.steps)-1),                       # half step in each other direction
                2*(len(self.steps)-1) * 2*len(self.steps))   # half step in each direction for every point in fine grid except halfway point

        m_steps = np.diag(self.steps)

        # Construct finer grid using half the previous step size in the direction of the faulty overlap
        kappas            = np.zeros((len(overlap_pairs), 1+dims[0], len(self.steps)))
        finer_grid        = np.zeros((len(overlap_pairs), 1+dims[0], len(self.steps)))
        if dims[0]>0:
            reference_grid  = np.zeros((len(overlap_pairs), dims[1], len(self.steps))) # these reference points do not correspond to actual grid points
        reference_grid_real = np.zeros((len(overlap_pairs),       2, len(self.steps))) # these are exactly the points between which the overlap was faulty

        for n,(i,j) in enumerate(overlap_pairs):
            # Take max of fully confined node kappas for the new finer nodes
            if isinstance(self.nodes[j],ReferenceNode):
                kappas[n,:] = self.nodes[i].kappas
            else:
                kappas[n,:] = np.max(np.array([self.nodes[i].kappas, self.nodes[j].kappas]),axis=0)

            # Determine direction of faulty overlap
            diff = np.abs(self.nodes[i].loc - self.nodes[j].loc)
            od = np.argmax(diff)
            reduced_m_steps = np.delete(m_steps,od,0)
            try:
                assert np.sum(np.hstack((diff[:od],diff[od+1:])))<precision
            except AssertionError:
                raise AssertionError('A problem occured when trying to define the overlap direction.')


            # Construct finer grid
            finer_grid[n,0] = 0.5*(self.nodes[i].loc + self.nodes[j].loc) # halfway point
            if dims[0]>0:
                finer_grid[n,1:] = finer_grid[n,0] + np.array([k*vec for k in [-0.5,0.5] for vec in reduced_m_steps]) # halfway step in every other direction

            # Construct real reference grid
            reference_grid_real[n][0] = self.nodes[i].loc
            reference_grid_real[n][1] = self.nodes[j].loc

            # Construct virtual reference_grid
            if dims[1]>0:
                reference_grid[n]  = np.array([fn + k*vec for fn in finer_grid[n,1:] for k in (-0.5,0.5) for vec in m_steps])

        kappas = kappas.reshape((-1,len(self.steps)))
        finer_grid = finer_grid.reshape((-1,len(self.steps)))
        if dims[0]>0:
            reference_grid = reference_grid.reshape((-1,len(self.steps)))
        reference_grid_real = reference_grid_real.reshape((-1,len(self.steps)))


        # Remove duplicate finer grid points, and store the corresponding kappa values
        mask = make_unique(self.steps/2.,finer_grid.reshape((-1,len(self.steps))),get_mask=True)
        finer_grid = np.round(finer_grid[mask],DECIMALS)
        kappas = kappas[mask]

        # Remove duplicate reference grid points
        if dims[1]>0:
            reference_grid  = np.round(make_unique(self.steps/2.,reference_grid),DECIMALS)
        reference_grid_real = np.round(make_unique(self.steps/2.,reference_grid_real),DECIMALS)

        # Remove reference grid points that correspond to finer grid points constructed above
        if dims[1]>0:
            reference_grid  = make_unique_between(self.steps/2.,finer_grid,reference_grid)
        reference_grid_real = make_unique_between(self.steps/2.,finer_grid,reference_grid_real) # Remove reference grid points that correspond to finer grid points constructed above\


        # Sort all the grids according to cvs
        ind = np.lexsort(finer_grid[:,::-1].T) # Sort by first column, then second column, ...
        finer_grid = finer_grid[ind]

        if dims[1]>0:
            ind = np.lexsort(reference_grid[:,::-1].T) # Sort by first column, then second column, ...
            reference_grid = reference_grid[ind]

        ind = np.lexsort(reference_grid_real[:,::-1].T) # Sort by first column, then second column, ...
        reference_grid_real = reference_grid_real[ind]

        # Convert to Nodes
        finer_nodes = []
        reference_nodes = []

        # Convert finer_grid points into FinerNodes
        finer_nodes = [FinerNode(point,kappas[n],MAX_KAPPA=self.MAX_KAPPA) for n,point in enumerate(finer_grid)]

        # Convert reference grid points to reference grid nodes
        if not self.major_grid is None:
            nodes = self.nodes + self.major_grid.nodes
        else:
            nodes = self.nodes

        tree = KDTree(np.array([node.loc for node in nodes]))

        # Add reference nodes to reference grid attribute
        # Check confinement of the reference nodes in new grid, if too low the overlap with them will never be good, calculate them too, except if they are already being refined!
        for node in reference_grid_real:
            ds,idx = tree.query(node,1) # find this point in grid
            if ds<10*precision:
                if isinstance(nodes[idx],Node) and nodes[idx].deviant:
                    # If this is a Node and it is deviant, we should wait to add it as a reference node
                    # since it does not make sense to test the overlap, and we do not want to start refining it already for the next grid
                    # It will be refined in the next iteration and will be taken into account when it is no longer deviant
                    #print("Not creating real reference node at {} because the trajectory data from {} is deviant.".format(node, nodes[idx].identity))
                    continue

                # Create a reference Node object, and check whether it is refined or should be refined
                rn = ReferenceNode(node, [nodes[idx]], real=True, refined=True)
                if self.traj_confinement(rn,factor=2.)<self.CONFINEMENT_THR:
                    rn.refine(self.KAPPA_GROWTH_FACTOR,self.MAX_KAPPA)
                reference_nodes.append(rn)


        if dims[1]>0:
            for node in reference_grid:
                ds,idx = tree.query(node,2) # find two closest neighbours
                neighbour_nodes = idx[(ds-ds[0])<precision] # check if they both fall inside the grid precision

                # Check whether the neighbouring nodes are sane and refined, if not, it does not make sense to create a reference yet
                # since we dont want to start a realization and refinement proces. It will be generated at a later instance
                include = True
                for nnode in neighbour_nodes:
                    if isinstance(nodes[nnode],Node) and nodes[nnode].deviant:
                        #print("Not creating virtual reference node at {} because the trajectory data from {} is deviant.".format(node, nodes[nnode].identity))
                        include = False
                        continue
                    if isinstance(nodes[nnode],ReferenceNode) and not nodes[nnode].refined:
                        #print("Not creating virtual reference node at {} because the reference trajectory data at {} is not refined.".format(node, nodes[nnode].loc))
                        include = False
                        continue

                # Caveat, virtual reference nodes will take trajectories of nodes, if a real reference node at that position would require refining (with the smaller cell size)
                # this virtual node will still be considered sane?    
                if include:
                    rn = ReferenceNode(node,[nodes[nnode] for nnode in neighbour_nodes], refined=True) # creates a virtual reference node, assume it is OK, it will get realized if not OK in overlap check
                    reference_nodes.append(rn)

        return finer_nodes, reference_nodes


    def refine_grid(self,verbose=False):
        """
            Make finer grid based on overlap heuristic and refine grid points for confinement.
            This also identifies a list of reference grid points required for this grid.
        """
        assert self.overlap is not None

        # Consider all points with no overlap for the finer grid
        if np.sum(1-self.overlap) > 0:
            finer_nodes, reference_nodes = self.refine_grid_nodes()
        else:
            finer_nodes, reference_nodes = [], []

        self.finer_nodes     = finer_nodes
        self.reference_nodes = reference_nodes

        if verbose:
            print('extreme_kappa:', [[l for l in n.loc] for n in self.nodes if isinstance(n,Node) and n.extreme_kappa])
            print('fine:', [[l for l in n.loc] for n in self.finer_nodes])
            print('ref:', [[l for l in n.loc] for n in self.reference_nodes])
            print('unrefined ref:', [[l for l in n.loc] for n in self.reference_nodes if not n.refined])
            