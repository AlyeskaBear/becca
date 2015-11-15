""" 
the Ziptie class 
"""
import numpy as np
import numba_tools as nb
import matplotlib.pyplot as plt    
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

    Attributes
    ----------
    activity_threshold : float
        Threshold below which input activity is teated as zero.
    agglomeration_energy : 2D array of floats
        The accumulated agglomeration energy for each bundle-cable pair.
    agglomeration_threshold
        Threshold above which agglomeration energy results in agglomeration.
    bundles_full : bool
        If True, all the bundles in the ``ZipTie`` are full and learning stops.
    bundle_activities : array of floats
        The current set of bundle activities.
    bundle_map_cols, bundle_map_rows : array of ints
        To represent the sparse 2D bundle map, a pair of row and col 
        arrays are used. Rows are bundle indices, and cols are 
        feature indices.  The bundle map shows which cables 
        are zipped together to form which bundles. 
    bundle_map_size : int
        The maximum number of non-zero entries in the bundle map.
    cable_activities : array of floats
        The current set of input actvities. 
    level : int
        The position of this ``ZipTie`` in the hierarchy. 
    max_num_bundles : int
        The maximum number of bundle outputs allowed.
    max_num_cables : int
        The maximum number of cable inputs allowed.
    name : str
        The name associated with the ``ZipTie``.
    nucleation_energy : 2D array of floats
        The accumualted nucleation energy associated with each cable-cable pair.
    nucleation_threshold : float
        Threshold above which nucleation energy results in nucleation.
    num_bundles : int
        The number of bundles that have been created so far.
    n_map_entries: int
        The total number of bundle map entries that have been created so far.
    size : int
        In this implementation, a characteristic of the ``ZipTie`` equal to
        ``max_num_cables`` and ``max_num_bundles``. 
    """

    def __init__(self, num_cables, name='anonymous', level=0):
        """ 
        Initialize each map, pre-allocating ``num_cables``.

        Parameters
        ----------
        level : int
            The position of this ``ZipTie`` in the hierarchy.
        num_cables : int
            The number of inputs to the ``ZipTie``.
        name : str
            The name assigned to the ``ZipTie``.
            Default is 'anonymous'.
        """
        self.name = name
        self.level = level
        self.max_num_cables = num_cables
        self.max_num_bundles = self.max_num_cables
        self.size = self.max_num_bundles
        self.num_bundles = 0
        # User-defined constants
        self.nucleation_threshold = 3e1
        self.agglomeration_threshold = .3 * self.nucleation_threshold
        self.activity_threshold = 1e-2

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
        
        This step combines the projection of cables activities
        to bundle activities together with using the cable activities 
        to incrementally train the ``ZipTie``.

        Parameters
        ----------
        modified_cables : list of ints
            The indices of the cable inputs that were modified in
            the prior ``ZipTie`` in the hierarchy. This can be used
            to treat those cables different, such as to reset any
            learning associated with them.
        new_cable_activities : array of floats
            The most recent set of cable activities.

        Returns
        -------
        agglomerated_bundle : int or None
            If not None, this is the index of the bundle that was agglomerated
            during this time step. Only one bundle may be agglomerated per
            time step, so this will never need to be a list.
        self.bundle_activities : array of floats
            The current activities of the bundles.
        modified_bundles : list of ints
            The indices of the bundels that were either nucleated or 
            agglomerated in this time step.
        """
        new_cable_activities = new_cable_activities.ravel()
        if new_cable_activities.size < self.max_num_cables:
            new_cable_activities = tools.pad(new_cable_activities, 
                                             self.max_num_cables)
        self.cable_activities = new_cable_activities

        # For any modified cables, reset their energy accumulation
        #if modified_cables is not None:
        # debug: don't prevent higher levels from using modified cables
        #if False:
        #    self.nucleation_energy[modified_cables,:] = 0.
        #    self.nucleation_energy[:,modified_cables] = 0.
        #    self.agglomeration_energy[:,modified_cables] = 0.

        #Find bundle activities by taking the minimum input value
        #in the set of cables in the bundle.
        threshold = np.max(self.cable_activities) * self.activity_threshold
        nb.sparsify_array1d_threshold(self.cable_activities, threshold)

        # Calculate ``bundle_energies``, an estimate of how much
        # the cables' activities contribute to each bundle. 
        self.bundle_energies = np.zeros(self.max_num_bundles)
        if self.n_map_entries > 0:
            # Call on numba throughout this method to do the heavy lifting.
            nb.sum_sparse_col_weights(
                    self.bundle_map_rows[:self.n_map_entries], 
                    self.bundle_map_cols[:self.n_map_entries], 
                    self.cable_activities, self.bundle_energies)

        # Calculate ``bundle_activities``, an estimate of how much
        # the cables' activities contribute to each bundle. 
        self.bundle_activities = np.zeros(self.max_num_bundles)
        self.bundle_activities[:self.num_bundles] = 1.
        if self.n_map_entries > 0:
            nb.min_sparse_col_weights(
                    self.bundle_map_rows[:self.n_map_entries], 
                    self.bundle_map_cols[:self.n_map_entries], 
                    self.cable_activities, self.bundle_activities)

        # Attempt to reconstruct the original ``cable_activities``
        # using the ``bundle_activities``.
        self.reconstruction = np.zeros(self.max_num_cables)
        if self.n_map_entries > 0:
            nb.max_sparse_row_weights(
                    self.bundle_map_rows[:self.n_map_entries], 
                    self.bundle_map_cols[:self.n_map_entries], 
                    self.bundle_activities, self.reconstruction)

        # Find the error, the portion of the original 
        # ``cable_activities`` that is not represented by the 
        # bundles' activities. This is what is used to train the ``ZipTie``.
        self.nonbundle_activities = self.cable_activities - self.reconstruction
        nb.sparsify_array1d_threshold(
                self.nonbundle_activities, self.activity_threshold)

        # As appropriate update the co-activity estimate and 
        # create new bundles.
        modified_cables = []
        nucleated_cables = None
        agglomerated_bundle = None
        agglomerated_cable = None
        if not self.bundles_full:
            nucleated_cables = self._create_new_bundles()
            agglomerated_cable , agglomerated_bundle = self._grow_bundles()
            # debug: Disable reporting modified bundles.
            #if nucleated_cables is not None:
            #    for cable in nucleated_cables:
            #        modified_cables.append(cable)
            #if agglomerated_cable is not None:
            #    modified_cables.append(agglomerated_cable)

        return self.bundle_activities, modified_cables, agglomerated_bundle

    def _create_new_bundles(self):
        """ 
        If the right conditions have been reached, create a new bundle.

        Returns
        -------
        nucleated_cables : list of ints or None
            If None, no bundles were nucleated.
            Otherwise, a two element list of indices of the 
            ``ZipTie``'s inputs that were nucleated into a new bundle.
        """
        # Incrementally accumulate nucleation energy.
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
        if max_energy > self.nucleation_threshold:
            nucleated_cables = [cable_index_a, cable_index_b]
            self.bundle_map_rows[self.n_map_entries] = self.num_bundles
            self.bundle_map_cols[self.n_map_entries] = cable_index_a
            self.increment_n_map_entries()
            self.bundle_map_rows[self.n_map_entries] = self.num_bundles
            self.bundle_map_cols[self.n_map_entries] = cable_index_b
            self.increment_n_map_entries()
            self.num_bundles += 1
            print ' '.join(['    ', self.name, 
                           'bundle', str(self.num_bundles), 
                           'added with', str(cable_index_a), 
                           str(cable_index_b)]) 

            # Check whether the ``ZipTie``'s capacity has been reached.
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
        Add one to ``n_map`` entries and grow the bundle map as needed.
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
        Update an estimate of co-activity between all cables.

        Returns
        -------
        agglomerated_bundle : int
            The index of the bundle to which a cable was agglomerated.
        agglomerated_cable : int
            The index of the cable that was agglomerated.
        """
        # Incrementally accumulate agglomeration energy.
        nb.agglomeration_energy_gather(self.bundle_activities, 
                                       self.nonbundle_activities,
                                       self.num_bundles,
                                       self.agglomeration_energy)

        # Don't accumulate agglomeration energy between cables already 
        # in the same bundle 
        val = 0.
        if self.n_map_entries > 0:
            print 'ae', self.agglomeration_energy 
            print 'bmr', self.bundle_map_rows[:self.n_map_entries] 
            print 'bmc', self.bundle_map_cols[:self.n_map_entries]
            print 'val', val
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
        if max_energy > self.agglomeration_threshold:
            agglomerated_bundle = bundle_index
            agglomerated_cable = cable_index
            self.bundle_map_rows[self.n_map_entries] = agglomerated_bundle
            self.bundle_map_cols[self.n_map_entries] = agglomerated_cable
            self.nucleation_energy[agglomerated_cable, :] = 0.
            self.nucleation_energy[:, agglomerated_cable] = 0.
            print ''.join(['            bundle grown', 
                          self.name, str(agglomerated_bundle), 
                          str(agglomerated_cable)])

        return agglomerated_cable, agglomerated_bundle
        
    def get_index_projection(self, bundle_index):
        """ 
        Project ``bundle_index`` down to its cable indices.

        Parameters
        ----------
        bundle_index : int
            The index of the bundle to project onto its constituent cables.

        Returns
        -------
        projection : array of floats
            An array of zeros and ones, representing all the cables that
            contribute to the bundle. The values ``projection``
            corresponding to all the cables that contribute are 1.
        """
        projection = np.zeros(self.max_num_cables)
        for i in range(self.n_map_entries):
            if self.bundle_map_rows[i] == bundle_index:
                projection[self.bundle_map_cols[i]] = 1.
        return projection
        
    def visualize(self):
        """
        Turn the state of teh ``ZipTie`` into an image.
        """
        print self.bundle_map_rows[:self.n_map_entries]
        print self.bundle_map_cols[:self.n_map_entries]

        if self.n_map_entries > 0:
            # Render the bundle map.
            bundle_map = np.zeros((self.max_num_cables, self.max_num_bundles))
            nb.set_dense_val(bundle_map, 
                             self.bundle_map_rows[:self.n_map_entries],
                             self.bundle_map_cols[:self.n_map_entries], 1.)
            tools.visualize_array(bundle_map, label=self.name + '_bundle_map')

            # Render the agglomeration energy.
            label = '_'.join([self.name, 'agg_energy'])
            tools.visualize_array(self.agglomeration_energy, label=label)
            plt.xlabel( str(np.max(self.agglomeration_energy)) )

            # Render the nucleation energy.
            label = '_'.join([self.name, 'nuc_energy'])
            tools.visualize_array(self.nucleation_energy, label=label)
            plt.xlabel( str(np.max(self.nucleation_energy)) )
