#!/usr/bin/env python
from __future__ import division
import numpy as np

from molmod.units import *
from molmod.constants import *

from yaff import *

#########################
# Potential class

class AbstractPotential(object):
    """Abstract potential class"""

    def __init__(self, cvs, kappas):
        self.energy = 0.
        self.pos = np.array(cvs)
        self.gpos = np.zeros_like(self.pos)
        self.ref = np.array(cvs)
        self.kappas = np.array(kappas)
        self.state = SimpleCVStateItem(self)

    def update_pos(self,pos):
        self.pos = pos

    def compute(self,gpos):
        my_gpos = self.gpos
        my_gpos[:] = 0.0

        # Umbrella contribution
        delta_cv = self.pos - self.ref

        my_gpos += self.kappas*delta_cv
        energy = 0.5*np.sum(self.kappas*delta_cv**2)

        # Underlying potential contribution
        energy += self.internal_compute(my_gpos)
        self.energy = energy


        if np.isnan(self.energy):
            raise ValueError('The energy is not-a-number (nan).')

        if gpos is not None:
            if np.isnan(my_gpos).any():
                raise ValueError('Some gpos element(s) is/are not-a-number (nan).')
            gpos += my_gpos
        return self.energy

    def internal_compute(self,gpos):
        """
            Implement this method in the potential.py file
        """
        raise NotImplementedError

    @staticmethod
    def eval(x):
        """
            Implement this method in the potential.py file
            - to compare with numerical FES
        """
        raise NotImplementedError


#########################
# Simple classes for analytic MD using Yaff Verlet code

class SimpleHDF5Writer(Hook):
    def __init__(self, f, start=0, step=1):
        """
           **Argument:**

           f
                A h5.File object to write the trajectory to.

           **Optional arguments:**

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.
        """
        self.f = f
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        if 'trajectory' not in self.f:
            self.init_trajectory(iterative)
        tgrp = self.f['trajectory']
        # determine the row to write the current iteration to. If a previous
        # iterations was not completely written, then the last row is reused.
        row = min(tgrp[key].shape[0] for key in iterative.state if key in tgrp.keys())
        for key, item in iterative.state.items():
            if item.value is None:
                continue
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            ds = tgrp[key]
            if ds.shape[0] <= row:
                # do not over-allocate. hdf5 works with chunks internally.
                ds.resize(row+1, axis=0)
            ds[row] = item.value


    def init_trajectory(self, iterative):
        tgrp = self.f.create_group('trajectory')
        for key, item in iterative.state.items():
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            if item.value is None:
                continue
            maxshape = (None,) + item.shape
            shape = (0,) + item.shape
            dset = tgrp.create_dataset(key, shape, maxshape=maxshape, dtype=item.dtype)
            for name, value in item.iter_attrs(iterative):
               tgrp.attrs[name] = value

class SimpleCVStateItem(StateItem):
    def __init__(self, potential):
        self.potential = potential
        StateItem.__init__(self, 'cv_values')

    def get_value(self, iterative):
        return self.potential.pos

class SimplePosStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'pos')

    def get_value(self, iterative):
        return iterative.potential.pos


class SimpleIterative(Iterative):
    default_state = []
    log_name = 'ITER'
    def __init__(self, potential, state=None, hooks=None, counter0=0, log=None, timer=None):
        """
           **Arguments:**

           potential
                The Potential instance used in the iterative algorithm

           **Optional arguments:**

           state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                iterative algorithm.

           hooks
                A function (or a list of functions) that is called after every
                iterative.

           counter0
                The counter value associated with the initial state.
        """
        self.potential = potential
        self.log = log
        self.timer = timer
        if state is None:
            self.state_list = [state_item.copy() for state_item in self.default_state]
        else:
            #self.state_list = state
            self.state_list = [state_item.copy() for state_item in self.default_state]
            self.state_list += state
        self.state = dict((item.key, item) for item in self.state_list)
        if hooks is None:
            self.hooks = []
        elif hasattr(hooks, '__len__'):
            self.hooks = hooks
        else:
            self.hooks = [hooks]
        self._add_default_hooks()
        self.counter0 = counter0
        self.counter = counter0
        with self.log.section(self.log_name), self.timer.section(self.log_name):
            self.initialize()

        # Initialize restart hook if present
        from yaff.sampling.io import RestartWriter
        for hook in self.hooks:
            if isinstance(hook, RestartWriter):
                hook.init_state(self)

    def initialize(self):
        self.call_hooks()

    def call_hooks(self):
        with self.timer.section('%s hooks' % self.log_name):
            state_updated = False
            from yaff.sampling.io import RestartWriter
            for hook in self.hooks:
                if hook.expects_call(self.counter) and not (isinstance(hook, RestartWriter) and self.counter==self.counter0):
                    if not state_updated:
                        for item in self.state_list:
                            item.update(self)
                        state_updated = True
                    if isinstance(hook, RestartWriter):
                        for item in hook.state_list:
                            item.update(self)
                    hook(self)

class SimpleConsErrTracker(object):
    '''
        A class that tracks the errors on the conserved quantity.
        Given its superior numerical accuracy, the algorithm below
        is used to calculate the running average. Its properties are discussed
        in Donald Knuth's Art of Computer Programming, vol. 2, p. 232, 3rd edition.
    '''
    def __init__(self, restart_h5=None):
        if restart_h5 is None:
            self.counter = 0
            self.ekin_m = 0.0
            self.ekin_s = 0.0
            self.econs_m = 0.0
            self.econs_s = 0.0
        else:
            tgrp = restart_h5['trajectory']
            self.counter = tgrp['counter'][-1]
            self.ekin_m = tgrp['ekin_m'][-1]
            self.ekin_s = tgrp['ekin_s'][-1]
            self.econs_m = tgrp['econs_m'][-1]
            self.econs_s = tgrp['econs_s'][-1]

    def update(self, ekin, econs):
        if self.counter == 0:
            self.ekin_m = ekin
            self.econs_m = econs
        else:
            ekin_tmp = ekin - self.ekin_m
            self.ekin_m += ekin_tmp/(self.counter+1)
            self.ekin_s += ekin_tmp*(ekin - self.ekin_m)
            econs_tmp = econs - self.econs_m
            self.econs_m += econs_tmp/(self.counter+1)
            self.econs_s += econs_tmp*(econs - self.econs_m)
        self.counter += 1

    def get(self):
        if self.counter > 1:
            # Returns the square root of the ratio of the
            # variance in Ekin to the variance in Econs
            return np.sqrt(self.econs_s/self.ekin_s)
        return 0.0


class SimpleVerletIntegrator(SimpleIterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('time'),
        AttributeStateItem('epot'),
        SimplePosStateItem(),
        AttributeStateItem('vel'),
        AttributeStateItem('rmsd_delta'),
        AttributeStateItem('rmsd_gpos'),
        AttributeStateItem('ekin'),
        TemperatureStateItem(),
        AttributeStateItem('etot'),
        AttributeStateItem('econs'),
        AttributeStateItem('cons_err'),
    ]
    log_name = 'VERLET'
    def __init__(self, potential, timestep=None, state=None, hooks=None, vel0=None,
                 temp0=300, scalevel0=True, time0=None, ndof=None, counter0=None, restart_h5=None, log=None, timer=None):
        """
            **Arguments:**

            potential
                An instance of the Potential class

            **Optional arguments:**

            timestep
                The integration time step (in atomic units)

            state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                iterative algorithm.

            hooks
                A function (or a list of functions) that is called after every
                iterative.

            vel0
                An array with initial velocities. If not given, random
                velocities are sampled from the Maxwell-Boltzmann distribution
                corresponding to the optional arguments temp0 and scalevel0

            temp0
                The (initial) temperature for the random initial velocities

            scalevel0
                If True (the default), the random velocities are rescaled such
                that the instantaneous temperature coincides with temp0.

            time0
                The time associated with the initial state.

            ndof
                When given, this option overrides the number of degrees of
                freedom determined from internal heuristics. When ndof is not
                given, its default value depends on the thermostat used. In most
                cases it is N. The ndof attribute is used to derive the
                temperature from the kinetic energy.

            counter0
                The counter value associated with the initial state.

            restart_h5
                HDF5 object containing the restart information
        """
        # Assign init arguments
        if timestep is None and restart_h5 is None:
            raise AssertionError('No Verlet timestep is found')
        self.ndof = ndof
        self.hooks = hooks
        self.restart_h5 = restart_h5
        self.log = log
        self.timer = timer

        # Retrieve the necessary properties if restarting. Restart objects
        # are overwritten by optional arguments in VerletIntegrator
        if self.restart_h5 is None:
            # set None variables to default value
            if time0 is None: time0 = 0.0
            if counter0 is None: counter0 = 0
            self.pos = potential.pos.copy()
            self.timestep = timestep
            self.time = time0
        else:
            # Arguments associated with the unit cell and positions are always retrieved
            tgrp = self.restart_h5['trajectory']
            self.pos = tgrp['pos'][-1,:,:]
            potential.update_pos(self.pos)

            # Arguments which can be provided in the VerletIntegrator object are only
            # taken from the restart file if not provided explicitly
            if time0 is None:
                self.time = tgrp['time'][-1]
            else:
                self.time = time0
            if counter0 is None:
                counter0 = tgrp['counter'][-1]
            if vel0 is None:
                vel0 = tgrp['vel'][-1,:,:]
            if timestep is None:
                self.timestep = self.restart_h5['/restart/timestep'][()]
            self._restart_add_hooks(self.restart_h5)

        # Verify the hooks: combine thermostat and barostat if present
        self._verify_hooks()

        # Set random initial velocities if needed.
        if vel0 is None:
            self.vel = self.get_random_vel(temp0, scalevel0, self.pos.shape)
        else:
            self.vel = vel0.copy()

        # Working arrays
        self.gpos = np.zeros(self.pos.shape, float)
        self.delta = np.zeros(self.pos.shape, float)

        # Tracks quality of the conserved quantity
        self._cons_err_tracker = SimpleConsErrTracker(restart_h5)
        SimpleIterative.__init__(self, potential, state, self.hooks, counter0, self.log, self.timer)

    def _add_default_hooks(self):
        if not any(isinstance(hook, VerletScreenLog) for hook in self.hooks):
            self.hooks.append(VerletScreenLog(log=self.log))

    def _verify_hooks(self):
        with self.log.section('ENSEM'):
            thermo = None
            index_thermo = 0

            # Look for the presence of a thermostat and/or barostat
            if hasattr(self.hooks, '__len__'):
                for index, hook in enumerate(self.hooks):
                    if hook.method == 'thermostat':
                        thermo = hook
                        index_thermo = index
            elif self.hooks is not None:
                if self.hooks.method == 'thermostat':
                    thermo = self.hooks

            if self.log.do_warning:
                if thermo is not None:
                    self.log('Temperature coupling achieved through ' + str(thermo.name) + ' thermostat')


    def _restart_add_hooks(self, restart_h5):
        # First, make sure that no thermostat hooks are supplied in the hooks argument.
        # If this is the case, they are NOT overwritten.
        thermo = None
        for hook in self.hooks:
            if hook.method == 'thermostat': thermo = hook

        if thermo is None: # not all hooks are already provided
            rgrp = restart_h5['/restart']
            tgrp = restart_h5['/trajectory']

            # verify if NHC thermostat is present
            if thermo is None and 'thermo_name' in rgrp:
                # collect thermostat properties and create thermostat
                thermo_name = rgrp['thermo_name'][()]
                timecon = rgrp['thermo_timecon'][()]
                temp = rgrp['thermo_temp'][()]

        # append the necessary hooks
        if thermo is not None:
            self.hooks.append(thermo)

    def initialize(self):
        # Standard initialization of Verlet algorithm
        self.gpos[:] = 0.0
        self.potential.update_pos(self.pos)
        self.epot = self.potential.compute(self.gpos)
        self.acc = -self.gpos
        self.posoud = self.pos.copy()

        # Allow for specialized initializations by the Verlet hooks.
        self.call_verlet_hooks('init')

        # Configure the number of degrees of freedom if needed
        if self.ndof is None:
            self.ndof = self.pos.size

        # Common post-processing of the initialization
        self.compute_properties(self.restart_h5)
        SimpleIterative.initialize(self) # Includes calls to conventional hooks

    def propagate(self):
        # Allow specialized hooks to modify the state before the regular verlet
        # step.
        self.call_verlet_hooks('pre')
        # Regular verlet step
        self.acc = -self.gpos
        self.vel += 0.5*self.acc*self.timestep
        self.pos += self.timestep*self.vel
        self.potential.update_pos(self.pos)
        self.gpos[:] = 0.0
        self.epot = self.potential.compute(self.gpos)
        self.acc = -self.gpos
        self.vel += 0.5*self.acc*self.timestep
        self.ekin = self._compute_ekin()

        # Allow specialized verlet hooks to modify the state after the step
        self.call_verlet_hooks('post')

        # Calculate the total position change
        self.posnieuw = self.pos.copy()
        self.delta[:] = self.posnieuw-self.posoud
        self.posoud[:] = self.posnieuw

        # Common post-processing of a single step
        self.time += self.timestep
        self.compute_properties()
        SimpleIterative.propagate(self) # Includes call to conventional hooks

    def _compute_ekin(self):
        '''Auxiliary routine to compute the kinetic energy

           This is used internally and often also by the Verlet hooks.
        '''
        return 0.5*(self.vel**2).sum()

    def compute_properties(self, restart_h5=None):
        self.rmsd_gpos = np.sqrt((self.gpos**2).mean())
        self.rmsd_delta = np.sqrt((self.delta**2).mean())
        self.ekin = self._compute_ekin()
        self.temp = self.ekin/self.ndof*2.0/boltzmann
        self.etot = self.ekin + self.epot
        self.econs = self.etot
        for hook in self.hooks:
            if isinstance(hook, VerletHook):
                self.econs += hook.econs_correction
        if restart_h5 is not None:
            self.econs = restart_h5['trajectory/econs'][-1]
        else:
            self._cons_err_tracker.update(self.ekin, self.econs)
        self.cons_err = self._cons_err_tracker.get()

    def finalize(self):
        if self.log.do_medium:
            self.log.hline()

    def call_verlet_hooks(self, kind):
        # In this call, the state items are not updated. The pre and post calls
        # of the verlet hooks can rely on the specific implementation of the
        # VerletIntegrator and need not to rely on the generic state item
        # interface.
        with self.timer.section('%s special hooks' % self.log_name):
            for hook in self.hooks:
                if isinstance(hook, VerletHook) and hook.expects_call(self.counter):
                    if kind == 'init':
                        hook.init(self)
                    elif kind == 'pre':
                        hook.pre(self)
                    elif kind == 'post':
                        hook.post(self)
                    else:
                        raise NotImplementedError

    @staticmethod
    def get_random_vel(temp0, scalevel0, shape):
        '''Generate random velocities using a Maxwell-Boltzmann distribution

           **Arguments:**

           temp0
                The temperature for the Maxwell-Boltzmann distribution.

           scalevel0
                When set to True, the velocities are rescaled such that the
                instantaneous temperature coincides with temp0.

           **Returns:** An (N) array with random velocities.
        '''
        result = np.random.normal(0, 1, shape)*np.sqrt(boltzmann*temp0)
        if scalevel0 and temp0 > 0:
            temp = (result**2).mean()/boltzmann
            scale = np.sqrt(temp0/temp)
            result *= scale
        return result

class SimpleCSVRThermostat(VerletHook):
    name = 'CSVR'
    kind = 'stochastic'
    method = 'thermostat'
    def __init__(self, temp, start=0, timecon=100*femtosecond):
        """
            This is an implementation of the CSVR thermostat. The equations are
            derived in:

                Bussi, G.; Donadio, D.; Parrinello, M. J. Chem. Phys. 2007,
                126, 014101

            The implementation (used here) is derived in

                Bussi, G.; Parrinello, M. Comput. Phys. Commun. 2008, 179, 26-29

           **Arguments:**

           temp
                The temperature of thermostat.

           **Optional arguments:**

           start
                The step at which the thermostat becomes active.

           timecon
                The time constant of the CSVR thermostat.
        """
        self.temp = temp
        self.timecon = timecon
        VerletHook.__init__(self, start, 1)

    def init(self, iterative):
        if iterative.ndof is None:
            iterative.ndof = iterative.pos.shape[0]
        self.kin = 0.5*iterative.ndof*boltzmann*self.temp

    def pre(self, iterative, G1_add = None):
        c = np.exp(-iterative.timestep/self.timecon)
        R = np.random.normal(0, 1)
        S = (np.random.normal(0, 1, iterative.ndof-1)**2).sum()
        iterative.ekin = iterative._compute_ekin()
        fact = (1-c)*self.kin/iterative.ndof/iterative.ekin
        alpha = np.sign(R+np.sqrt(c/fact))*np.sqrt(c + (S+R**2)*fact + 2*R*np.sqrt(c*fact))
        iterative.vel[:] = alpha*iterative.vel
        iterative.ekin_new = alpha**2*iterative.ekin
        self.econs_correction += (1-alpha**2)*iterative.ekin
        iterative.ekin = iterative.ekin_new


    def post(self, iterative, G1_add = None):
        pass
