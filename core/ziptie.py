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
    """
    def __init__(self, min_cables, exploit=False, classifier=False,
                 name='anonymous', level=0):
        """ 
        Initialize each map, pre-allocating max_num_bundles 
        """
        self.exploit = exploit
        self.name = name
        self.level = level
        self.max_num_cables = int(2 ** np.ceil(np.log2(min_cables)))
        self.max_num_bundles = self.max_num_cables
        self.num_bundles = 0
        # User-defined constants
        self.NUCLEATION_THRESHOLD = 1e-2
        #self.NUCLEATION_ENERGY_RATE = 2e-5 
        self.NUCLEATION_ENERGY_RATE = (
                1e-4 * self.NUCLEATION_THRESHOLD  * 2 ** -level)
        self.NUCLEATION_ENERGY_DECAY_RATE = 1e-2 * self.NUCLEATION_ENERGY_RATE
        self.AGGLOMERATION_THRESHOLD = self.NUCLEATION_THRESHOLD
        self.AGGLOMERATION_ENERGY_RATE = self.NUCLEATION_ENERGY_RATE 
        self.AGGLOMERATION_ENERGY_DECAY_RATE = self.NUCLEATION_ENERGY_DECAY_RATE
        self.ACTIVITY_THRESHOLD = 1e-2
        #self.AGGLOMERATION_ENERGY_RATE = 1e-4 * 2**-level
        # The rate at which cable activities decay (float, 0 < x < 1)
        if classifier:
            self.ACTIVITY_DECAY_RATE = 1.
        else:
            self.ACTIVITY_DECAY_RATE = 2 ** (-self.level)

        self.bundles_full = False        
        self.bundle_activities = np.zeros(self.max_num_bundles)
        self.cable_activities = np.zeros(self.max_num_cables)
        map_size = (self.max_num_bundles, self.max_num_cables)
        #self.bundle_map = np.zeros(map_size)
        #self.bundle_map = np.ones(map_size) * np.nan
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
        self.cable_activities = tools.bounded_sum2(new_cable_activities, 
                self.cable_activities * (1. - self.ACTIVITY_DECAY_RATE))
        # For any modified cables, reset their energy accumulation
        if modified_cables is not None:
            self.nucleation_energy[modified_cables,:] = 0.
            self.nucleation_energy[:,modified_cables] = 0.
            self.agglomeration_energy[:,modified_cables] = 0.
        """
        Find bundle activities by taking the minimum input value
        in the set of cables in the bundle.
        """
        #self.bundle_map[np.where(self.bundle_map==0.)] = np.nan
        # Remove small values of cable activities.
        #ca_ind = np.where(self.cable_activities < self.ACTIVITY_THRESHOLD) 
        #print 'pre', self.cable_activities[
        #        np.where(self.cable_activities > 0.)].size
        threshold = np.max(self.cable_activities) * self.ACTIVITY_THRESHOLD
        nb.sparsify_array1d_threshold(self.cable_activities, threshold)

        #print 'post', self.cable_activities[
        #        np.where(self.cable_activities > 0.)].size

        #bundle_components = nb.multiply_sparse2d_rows(
        #        self.bundle_map_rows, self.bundle_map_cols, 
        #        self.cable_activities)
        self.bundle_energies = np.zeros(self.max_num_bundles)
        if self.n_map_entries > 0:
            nb.sum_sparse_col_weights(
                    self.bundle_map_rows[:self.n_map_entries], 
                    self.bundle_map_cols[:self.n_map_entries], 
                    self.cable_activities, self.bundle_energies)
        #print 'post', self.bundle_energies[
        #        np.where(self.bundle_energies > 0.)]

        #bundle_components = (self.bundle_map[:,ca_ind[0]] * 
        #                     self.cable_activities[ca_ind[0]].T 
        # TODO: consider other ways to calculate bundle energies
        #self.bundle_energies = np.sum(bundle_components, axis=1)[:,np.newaxis]
        #self.bundle_energies = np.nansum(bundle_components,
        #                                 axis=1)[:,np.newaxis]

        #self.bundle_energies[np.where(np.isnan(self.bundle_energies))] = 0.
        #print 'be', self.bundle_energies.T
        #energy_index = np.argsort(self.bundle_energies.ravel())[:,np.newaxis]
        energy_index = np.argsort(self.bundle_energies)
        #print 'ei', energy_index.T
        max_energy = self.bundle_energies[energy_index[-1]]
        #print 'max_e', max_energy
        mod_energies = self.bundle_energies - (
                max_energy * ( (self.num_bundles - energy_index) / 
                               (self.num_bundles + tools.EPSILON) ))
        #print 'me', mod_energies.T
        
        #self.bundle_activities = np.nanmin(bundle_components,
        #                                   axis=1)[:,np.newaxis]
        self.bundle_activities = np.zeros(self.max_num_bundles)
        self.bundle_activities[:self.num_bundles] = 1.
        if self.n_map_entries > 0:
            nb.min_sparse_col_weights(
                    self.bundle_map_rows[:self.n_map_entries], 
                    self.bundle_map_cols[:self.n_map_entries], 
                    self.cable_activities, self.bundle_activities)
        #print 'post', self.bundle_energies[
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
            '''
            print 'bmr', self.bundle_map_rows[:self.n_map_entries] 
            print 'bmc', self.bundle_map_cols[:self.n_map_entries]
            print 'ba', self.bundle_activities
            print 'recon', self.reconstruction
            print 'ca', self.cable_activities
            '''

        self.nonbundle_activities = self.cable_activities - self.reconstruction
        nb.sparsify_array1d_threshold(
                self.nonbundle_activities, self.ACTIVITY_THRESHOLD)
        #print 'recon', self.reconstruction.T
        #print 'nba', self.nonbundle_activities.T
        # As appropriate update the co-activity estimate and 
        # create new bundles
        if not self.exploit:
            modified_cables = []
            nucleated_cables = None
            if not self.bundles_full:
                nucleated_cables = self._create_new_bundles()
            agglomerated_bundle = None
            agglomerated_cable = None
            agglomerated_cable , agglomerated_bundle = self._grow_bundles()
            if nucleated_cables is not None:
                for cable in nucleated_cables:
                    modified_cables.append(cable)
            if agglomerated_cable is not None:
                modified_cables.append(agglomerated_cable)
        else:
            agglomerated_bundle = None

        #print 'agg', self.agglomeration_energy[np.where(self.agglomeration_energy > 0.)]
        #print 'bund', self.bundle_activities[np.where(self.bundle_activities > 0.)]
        #print 'mod', mod_energies
        #print 'mod > 0', mod_energies[np.where(mod_energies > 0.)]
        return self.bundle_activities, modified_cables, agglomerated_bundle

    def _create_new_bundles(self):
        """ 
        If the right conditions have been reached, create a new bundle 
        """
        # Bundle space is a scarce resource. Decay the energy.        
        #self.nucleation_energy -= (self.cable_activities *
        #                           self.cable_activities.T *
        #                           self.nucleation_energy * 
        #                           self.NUCLEATION_ENERGY_RATE)
        #nb.nucleation_energy_decay(self.cable_activities,
        #                           self.nucleation_energy, 
        #                           self.NUCLEATION_ENERGY_DECAY_RATE)
        #self.nucleation_energy += (self.nonbundle_activities * 
        #                           self.nonbundle_activities.T * 
        #                           #(1. - self.nucleation_energy) *
        #                           self.NUCLEATION_ENERGY_RATE) 
        nb.nucleation_energy_gather(self.nonbundle_activities,
                                    self.nucleation_energy, 
                                    self.NUCLEATION_ENERGY_RATE)
   
        #print self.nucleation_energy[np.where(self.nucleation_energy > 0.)]
        #print self.nucleation_energy[:10,:10]
        #print 'ca', np.where(self.cable_activities > 0.)
        #print 'nba', np.where(self.nonbundle_activities > 0.)
        #print np.max(self.nucleation_energy)
        #print 'ba', self.bundle_activities
        # Don't accumulate nucleation energy between a cable and itself
        ind = np.arange(self.cable_activities.size).astype(int)
        self.nucleation_energy[ind,ind] = 0.

        #max_energy = 0.
        #cable_index_a = 0
        #cable_index_b = 0
        results = -np.ones(3)
        nb.max_dense(self.nucleation_energy, results)
        max_energy = results[0]
        cable_index_a = int(results[1])
        cable_index_b = int(results[2])
        #print 'max', max_energy, 'a', cable_index_a, 'b', cable_index_b

        '''
        cable_indices = np.where(self.nucleation_energy > 
                                 self.NUCLEATION_THRESHOLD)
        #print 'ne'
        #print self.nucleation_energy
        # Add a new bundle if appropriate
        nucleated_cables = None
        if cable_indices[0].size > 0:
            # Identify the newly created bundle's cables.
            # Randomly pick a new cable from the candidates, 
            # if there is more than one
            pair_index = np.random.randint(cable_indices[0].size)
            cable_index_a = cable_indices[0][pair_index]
            cable_index_b = cable_indices[1][pair_index]
            nucleated_cables = [cable_index_a, cable_index_b]
            #cable_index = cable_indices[0][int(np.random.random_sample() * 
            #                               cable_indices[0].size)]
            # Create the new bundle in the bundle map.
            #self.bundle_map[self.num_bundles, cable_index] = 1.
            self.bundle_map[self.num_bundles, cable_index_a] = 1.
            self.bundle_map[self.num_bundles, cable_index_b] = 1.
        '''
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
            #self.nucleation_energy[cable_index, 0] = 0.
            #self.agglomeration_energy[:, cable_index] = 0.
            
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
        #import time
        #tic = time.clock()
        #for _ in range(100):
        #toc = time.clock()
        #print 'time', toc - tic
        #coactivities = np.dot(self.bundle_activities, 
        #                      self.nonbundle_activities.T)
        # 2.7-3.0 sec
        #print self.name
        #print 'ca', np.nonzero(coactivities)
        #print 'ba', self.bundle_activities.T
        #print 'nba', self.nonbundle_activities.T

        # Each cable's nonbundle activity is distributed to 
        # agglomeration energy with each bundle proportionally 
        # to their coactivities.
        # Decay the energy        
        # 3.3 sec
        #self.agglomeration_energy -= (self.cable_activities.T *
        #                              self.agglomeration_energy * 
        #                              self.AGGLOMERATION_ENERGY_RATE)
        nb.agglomeration_energy_decay(self.cable_activities, self.num_bundles,
                                      self.agglomeration_energy, 
                                      self.AGGLOMERATION_ENERGY_DECAY_RATE)
        # 3.3-3.8 sec
        #coactivities = self.bundle_activities * self.nonbundle_activities.T
        #self.agglomeration_energy += (coactivities * 
        #                              #(1. - self.agglomeration_energy) *
        #                              self.AGGLOMERATION_ENERGY_RATE)
        nb.agglomeration_energy_gather(self.bundle_activities, 
                                       self.nonbundle_activities,
                                       self.num_bundles,
                                       self.agglomeration_energy, 
                                       self.AGGLOMERATION_ENERGY_RATE)
        #print 'ba', self.bundle_activities[np.where(self.bundle_activities > 0.)]
        #print 'nab', self.nonbundle_activities[np.where(self.nonbundle_activities > 0.)]
        #print 'ae', np.sort(self.agglomeration_energy[np.where(self.agglomeration_energy > 0.)])

        #print self.name
        #print 'ae', np.sort(self.agglomeration_energy[np.nonzero(np.nan_to_num(
        #        self.agglomeration_energy))].ravel())

        # Don't accumulate agglomeration energy between cables already 
        # in the same bundle 
        # 1.9 sec
        #self.agglomeration_energy *= 1 - np.nan_to_num(self.bundle_map)
        val = 0.
        if self.n_map_entries > 0:
            nb.set_dense_val(self.agglomeration_energy, 
                             self.bundle_map_rows[:self.n_map_entries], 
                             self.bundle_map_cols[:self.n_map_entries], 
                             val)
        #self.agglomeration_energy *= 1 - self.bundle_map
        #max_energy = 0.
        #cable_index = 0
        #bundle_index = 0
        results = -np.ones(3)
        #nb.max_dense(self.agglomeration_energy, cable_index, 
        #             bundle_index, max_energy)
        nb.max_dense(self.agglomeration_energy, results)
        max_energy = results[0]
        cable_index = int(results[2])
        bundle_index = int(results[1])
        '''
        # 8.5-9.7 sec
        new_candidates = np.where(self.agglomeration_energy >= 
                                  self.AGGLOMERATION_THRESHOLD)
        num_candidates =  new_candidates[0].size 
        
        agglomerated_cable = None
        agglomerated_bundle = None
        if num_candidates > 0:
            candidate_index = np.random.randint(num_candidates) 
            agglomerated_cable = new_candidates[1][candidate_index]
            agglomerated_bundle = new_candidates[0][candidate_index]
        '''
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
            #self.agglomeration_energy[:, agglomerated_cable] = 0.
            print '            bundle grown', self.name, agglomerated_bundle, agglomerated_cable
        return agglomerated_cable, agglomerated_bundle
        
    def get_index_projection(self, bundle_index):
        """ 
        Project bundle indices down to their cable indices 
        """
        #projection = copy.deepcopy(self.bundle_map[bundle_index,:])
        #projection[np.isnan(projection)] = 0.
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
            #label = '_'.join([self.name, 'agg_energy', 
            #                  str(np.max(self.agglomeration_energy))])
            label = '_'.join([self.name, 'agg_energy'])
            tools.visualize_array(self.agglomeration_energy, label=label)
            plt.xlabel( str(np.max(self.agglomeration_energy)) )
            #label = '_'.join([self.name, 'nuc_energy', 
            #                  str(np.max(self.nucleation_energy))])
            label = '_'.join([self.name, 'nuc_energy'])
            tools.visualize_array(self.nucleation_energy, label=label)
            plt.xlabel( str(np.max(self.nucleation_energy)) )
        #import matplotlib.pyplot as plt    
        #plt.show()


