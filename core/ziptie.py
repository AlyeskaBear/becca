""" 
the Ziptie class 
"""
import copy
import numpy as np
import numba_tools as nb
import tools

class ZipTie(object):
    """ 
    An incremental unsupervised clustering algorithm

    Input channels are clustered together into mutually co-active sets.
    A helpful metaphor is bundling cables together with zip ties.
    Cables that carry related signals are commonly co-active, 
    and are grouped together. Cables that are co-active with existing
    bundles can be added to those bundles. A single cable may be ziptied
    into several different bundles. Co-activity is estimated 
    incrementally, that is, the algorithm updates the estimate after 
    each new set of signals is received. 

    Zipties are arranged hierarchically within the agent's drivetrain. 
    The agent begins with only one ziptie and creates subsequent
    zipties as previous ones mature. 

    When stacked within a gearbox, 
    zipties form a sparse deep neural network (DNN). 
    This DNN has the extremely desirable characteristic of
    l-0 sparsity--the number of non-zero weights are minimized.
    The vast majority of weights in this network are zero,
    and the rest are one.
    This makes sparse computation feasible and allows for 
    straightforward interpretation and visualization of the
    features at each level.  
    """
    def __init__(self, min_cables, exploit=False, classifier=False,
                 name='anonymous', level=0):
        """ 
        Initialize each map, pre-allocating max_num_bundles 
        """
        self.exploit = exploit
        self.name = name
        self.level = level
        self.max_num_cables = min_cables
        self.max_num_bundles = self.max_num_cables
        self.num_bundles = 0
        # User-defined constants#
        # was 
        #self.NUCLEATION_THRESHOLD = 1e-2 * 2**(-level)
        # higher level features too basic, grew to slowly
        #self.NUCLEATION_THRESHOLD = 1e-2 * 4**(-level)
        self.NUCLEATION_THRESHOLD = 1e3 #* 2 ** (-level)
        #self.NUCLEATION_ENERGY_RATE = 1e-3 * self.NUCLEATION_THRESHOLD
        # TODO: remove this 
        #self.NUCLEATION_ENERGY_RATE = 1.
        self.AGGLOMERATION_THRESHOLD = self.NUCLEATION_THRESHOLD
        # TODO: remove this 
        #self.AGGLOMERATION_ENERGY_RATE = self.NUCLEATION_ENERGY_RATE
        self.ACTIVITY_THRESHOLD = 1e-2
        # The rate at which cable activities decay (float, 0 < x < 1)
        # TODO: remove activity decay rate
        if classifier:
            self.ACTIVITY_DECAY_RATE = 1.
        else:
            #self.ACTIVITY_DECAY_RATE = 2 ** (-self.level)
            self.ACTIVITY_DECAY_RATE = 1.

        self.bundles_full = False        
        self.bundle_activities = np.zeros(self.max_num_bundles)
        self.cable_activities = np.zeros(self.max_num_cables)
        map_size = (self.max_num_bundles, self.max_num_cables)
        # To represent the sparse 2D bundle map, a pair of row and col 
        # arrays are used. Rows are bundle indices, and cols are 
        # feature indices.
        self.bundle_map_size = 8 
        self.bundle_map_rows = -np.ones(self.bundle_map_size) 
        self.bundle_map_cols = -np.ones(self.bundle_map_size) 
        self.n_map_entries = 0
        self.agglomeration_energy = np.zeros(map_size)
        self.nucleation_energy = np.zeros((self.max_num_cables, 
                                           self.max_num_cables))

    def step_up(self, new_cable_activities, modified_cables):
        """ 
        Update co-activity estimates and calculate bundle activity 
        """
        new_cable_activities = new_cable_activities.ravel()
        if new_cable_activities.size < self.max_num_cables:
            new_cable_activities = tools.pad(new_cable_activities, 
                                             self.max_num_cables)
        # debug: don't adapt cable activities 
        self.cable_activities = new_cable_activities
        #self.cable_activities = tools.bounded_sum2(new_cable_activities, 
        #        self.cable_activities * (1. - self.ACTIVITY_DECAY_RATE))
        # For any modified cables, reset their energy accumulation
        #if modified_cables is not None:
        # debug: don't prevent higher levels from using modified cables
        #if False:
        #    self.nucleation_energy[modified_cables,:] = 0.
        #    self.nucleation_energy[:,modified_cables] = 0.
        #    self.agglomeration_energy[:,modified_cables] = 0.
        """
        Find bundle activities by taking the minimum input value
        in the set of cables in the bundle.
        """
        threshold = (np.max(self.cable_activities) * self.ACTIVITY_THRESHOLD + 
                     tools.EPSILON)
        nb.sparsify_array1d_threshold(self.cable_activities, threshold)
        '''
        self.bundle_energies = np.zeros(self.max_num_bundles)
        if self.n_map_entries > 0:
            nb.sum_sparse_col_weights(
                    self.bundle_map_rows[:self.n_map_entries], 
                    self.bundle_map_cols[:self.n_map_entries], 
                    self.cable_activities, self.bundle_energies)
        energy_index = np.argsort(self.bundle_energies)
        max_energy = self.bundle_energies[energy_index[-1]]
        mod_energies = self.bundle_energies - (
                max_energy * ( (self.num_bundles - energy_index) / 
                               (self.num_bundles + tools.EPSILON) ))
        '''
        self.bundle_activities = np.zeros(self.max_num_bundles)
        # Initialize bundle activities to the highest possible minimum value
        self.bundle_activities[:self.num_bundles] = 1.
        if self.n_map_entries > 0:
            nb.min_sparse_col_weights(
                    self.bundle_map_rows[:self.n_map_entries], 
                    self.bundle_map_cols[:self.n_map_entries], 
                    self.cable_activities, self.bundle_activities)
        # Debug Go with average activation
        #self.bundle_activities = np.min(bundle_components, axis=1)[:,np.newaxis]
        #self.bundle_activities = self.bundle_energies / (
        #        np.sum(self.bundle_map, axis=1)[:,np.newaxis] + tools.EPSILON)
        #print 'ba before', self.bundle_activities.T
        # debug
        # TODO: turn this off and test
        #self.bundle_activities[np.where(mod_energies < 0.)] = 0.

        #self.bundle_activities = np.nan_to_num(self.bundle_activities)
        #self.bundle_activities[np.where(np.isnan(self.bundle_activities))] = 0.
        #print 'ba', self.bundle_activities.T

        #self.reconstruction = np.nanmax(self.bundle_activities * 
        #                                self.bundle_map, 
        #                                axis=0)[:, np.newaxis]
        #self.reconstruction = np.max(self.bundle_activities * 
        #                             self.bundle_map, 
        #                             axis=0)[:, np.newaxis]
        #self.reconstruction[np.isnan(self.reconstruction)] = 0.
        self.reconstruction = np.zeros(self.max_num_cables)
        if self.n_map_entries > 0:
            nb.max_sparse_row_weights(
                    self.bundle_map_rows[:self.n_map_entries], 
                    self.bundle_map_cols[:self.n_map_entries], 
                    self.bundle_activities, self.reconstruction)

        self.nonbundle_activities = self.cable_activities - self.reconstruction
        nb.sparsify_array1d_threshold(
                self.nonbundle_activities, self.ACTIVITY_THRESHOLD)
        # As appropriate update the co-activity estimate and 
        # create new bundles
        if not self.exploit:
            modified_cables = []
            nucleated_cables = None
            agglomerated_bundle = None
            agglomerated_cable = None
            if not self.bundles_full:
                #nucleated_cables = self._create_new_bundles()
                #agglomerated_cable , agglomerated_bundle = self._grow_bundles()
                if nucleated_cables is not None:
                    for cable in nucleated_cables:
                        modified_cables.append(cable)
                if agglomerated_cable is not None:
                    modified_cables.append(agglomerated_cable)
        else:
            agglomerated_bundle = None
        # debug
        self.bundle_activities = np.zeros(self.bundle_activities.shape)
        return self.bundle_activities, modified_cables, agglomerated_bundle

    def _create_new_bundles(self):
        """ 
        If the right conditions have been reached, create a new bundle 
        """
        # Bundle space is a scarce resource. Decay the energy.        
        nb.nucleation_energy_gather(self.nonbundle_activities,
                                    self.nucleation_energy)
   
        # Don't accumulate nucleation energy between a cable and itself
        ind = np.arange(self.cable_activities.size).astype(int)
        self.nucleation_energy[ind,ind] = 0.
        results = -np.ones(3)
        nb.max_dense(self.nucleation_energy, results)
        max_energy = results[0]
        cable_index_a = int(results[1])
        cable_index_b = int(results[2])
        # Add a new bundle if appropriate
        nucleated_cables = None
        if max_energy > self.NUCLEATION_THRESHOLD:
            nucleated_cables = [cable_index_a, cable_index_b]
            self.bundle_map_rows[self.n_map_entries] = self.num_bundles
            self.bundle_map_cols[self.n_map_entries] = cable_index_a
            self.increment_n_map_entries()
            self.bundle_map_rows[self.n_map_entries] = self.num_bundles
            self.bundle_map_cols[self.n_map_entries] = cable_index_b
            self.increment_n_map_entries()
            self.num_bundles += 1
            print '    ', self.name, 'bundle', self.num_bundles, 'added with', cable_index_a, cable_index_b 
            if self.num_bundles == self.max_num_bundles:
                self.bundles_full = True
            # Reset the accumulated nucleation and agglomeration energy
            # for the two cables involved.
            self.nucleation_energy[cable_index_a, :] = 0.
            self.nucleation_energy[cable_index_b, :] = 0.
            self.nucleation_energy[:, cable_index_a] = 0.
            self.nucleation_energy[:, cable_index_b] = 0.
            #self.agglomeration_energy[:, cable_index_a] = 0.
            #self.agglomeration_energy[:, cable_index_b] = 0.
        return nucleated_cables

    def increment_n_map_entries(self):
        """
        Add one to n_map entries and grow the bundle map as needed
        """
        self.n_map_entries += 1
        if self.n_map_entries >= self.bundle_map_size:
            self.bundle_map_size *= 2
            self.bundle_map_rows = tools.pad(self.bundle_map_rows, 
                                             self.bundle_map_size, val=-1.)
            self.bundle_map_cols = tools.pad(self.bundle_map_cols, 
                                             self.bundle_map_size, val=-1.)
        
    def _grow_bundles(self):
        """ 
        Update an estimate of co-activity between all cables 
        """
        nb.agglomeration_energy_gather(self.bundle_activities, 
                                       self.nonbundle_activities,
                                       self.num_bundles,
                                       self.agglomeration_energy)

        # Don't accumulate agglomeration energy between cables already 
        # in the same bundle 
        val = 0.
        if self.n_map_entries > 0:
            nb.set_dense_val(self.agglomeration_energy, 
                             self.bundle_map_rows[:self.n_map_entries], 
                             self.bundle_map_cols[:self.n_map_entries], 
                             val)
        results = -np.ones(3)
        nb.max_dense(self.agglomeration_energy, results)
        max_energy = results[0]
        cable_index = int(results[2])
        bundle_index = int(results[1])
        # Add a cable to a bundle if appropriate
        agglomerated_cable = None
        agglomerated_bundle = None
        if max_energy > self.AGGLOMERATION_THRESHOLD:
            agglomerated_bundle = bundle_index
            agglomerated_cable = cable_index
            #self.bundle_map[agglomerated_bundle, agglomerated_cable] = 1.
            self.bundle_map_rows[self.n_map_entries] = agglomerated_bundle
            self.bundle_map_cols[self.n_map_entries] = agglomerated_cable
            self.increment_n_map_entries()
            #self.nucleation_energy[agglomerated_cable, 0] = 0.
            self.nucleation_energy[agglomerated_cable, :] = 0.
            self.nucleation_energy[:, agglomerated_cable] = 0.
            print '            bundle grown', self.name, agglomerated_bundle, agglomerated_cable
        return agglomerated_cable, agglomerated_bundle
        
    def get_index_projection(self, bundle_index):
        """ 
        Project bundle indices down to their cable indices 
        """
        projection = np.zeros(self.max_num_cables)
        for i in range(self.n_map_entries):
            if self.bundle_map_rows[i] == bundle_index:
                projection[self.bundle_map_cols[i]] = 1.
        return projection
        
    def bundles_created(self):
        return self.num_bundles

    def visualize(self, save_eps=False):
        import matplotlib.pyplot as plt    
        print self.bundle_map_rows[:self.n_map_entries]
        print self.bundle_map_cols[:self.n_map_entries]
        bundle_map = np.zeros((self.max_num_cables, self.max_num_bundles))
        if self.n_map_entries > 0:
            nb.set_dense_val(bundle_map, 
                             self.bundle_map_rows[:self.n_map_entries],
                             self.bundle_map_cols[:self.n_map_entries], 1.)
            tools.visualize_array(bundle_map, label=self.name + '_bundle_map')
            label = '_'.join([self.name, 'agg_energy'])
            tools.visualize_array(self.agglomeration_energy, label=label)
            plt.xlabel( str(np.max(self.agglomeration_energy)) )
            label = '_'.join([self.name, 'nuc_energy'])
            tools.visualize_array(self.nucleation_energy, label=label)
            plt.xlabel( str(np.max(self.nucleation_energy)) )
