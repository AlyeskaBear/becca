""" 
the Ziptie class 
"""
from numba import jit
import numpy as np
import matplotlib.pyplot as plt    
import tools

class ZipTie(object):
    """ 
    An incremental unsupervised clustering algorithm.

    Input channels are clustered together into mutually co-active sets.
    A helpful metaphor is bundling cables together with zip ties.
    Cables that carry related signals are commonly co-active, 
    and are grouped together. Cables that are co-active with existing
    bundles can be added to those bundles. A single cable may be ziptied
    into several different bundles. Co-activity is estimated 
    incrementally, that is, the algorithm updates the estimate after 
    each new set of signals is received. 

    Zipties can be stacked hierarchically, with the bundles of one
    serving as the cables for the next.
    When stacked, zipties form a sparse deep neural network (DNN). 
    This DNN has the desirable characteristic of
    l-0 sparsity--the number of non-zero weights are minimized.
    The vast majority of weights in this network are zero,
    and the rest are one.
    This makes sparse computation feasible and allows for 
    straightforward interpretation and visualization of the
    features at each level.  

    Attributes
    ----------
    agglomeration_energy : 2D array of floats
        The accumulated agglomeration energy for each bundle-cable pair.
    agglomeration_threshold
        Threshold above which agglomeration energy results in agglomeration.
    bundles_full : array of int
        Element i of the array represents he "full" status of 
        bundle i. Indicates whether a 
        sequence can accept additional elements. 1 if full, 0 if not.
    full : bool
        If True, all the bundles in the ``ZipTie`` are full and learning stops.
    max_num_bundles : int
        The maximum number of bundles (outputs) allowed.
    max_num_cables : int
        The maximum number of cables (inputs) allowed.
    members : 2D array of int
        The membership map from cables to bundles. Each row represents
        one bundle. It contains the indices of cables that contribute
        to it. Empty cables in the array are filled with -1.
    name : str
        The name associated with this ``ZipTie``.
    nucleation_energy : 2D array of floats
        The accumualted nucleation energy associated with each cable-cable pair.
    nucleation_threshold : float
        Threshold above which nucleation energy results in nucleation.
    num_bundles : int
        The number of bundles that have been created so far.
    num_members : array of int
        The number of cables contributing to each bundle.
        Element i corresponds to bundle i.
    """

    def __init__(self, num_cables, num_bundles=None, 
                 name=None, speed=0):
        """ 
        Initialize the ``ZipTie``, pre-allocating ``num_cables``.

        Parameters
        ----------
        speed : int
            A control on how fast  the ``ZipTie`` learns.
            0 is recommended if it is standalone. 1 is slower 
            and 2 is slower still.
            If used in a hierarchy, assign a speed of i to the ith level.
        num_bundles : int
            The number of outputs from the ``ZipTie``.
        num_cables : int
            The number of inputs to the ``ZipTie``.
        name : str
            The name assigned to the ``ZipTie``.
            Default is 'anonymous'.
        """
        if name is None:
            name = '_'.join(['ziptie', str(speed)])
        else:
            self.name = name

        self.max_num_cables = num_cables
        if num_bundles is None:
            self.max_num_bundles = self.max_num_cables
        else:
            self.max_num_bundles = num_bundles
        self.num_bundles = 0


        self.full = False        
        self.max_cables_per_bundle = 10

        # Pre-allocate the membership map from cables to bundles. 
        self.members = -1 * np.ones((self.max_num_bundles,
                                     self.max_cables_per_bundle)).astype(int)
        self.bundles_full = np.zeros(self.max_num_bundles).astype(int)
        self.num_members = np.zeros(self.max_num_bundles).astype(int)

        # Thresholds determine how long it takes for bundle nucleation
        # and agglomeration to occur. Higher thresholds take longer
        # to reach.
        self.nucleation_threshold = 100. * 5 ** speed
        self.agglomeration_threshold = .5 * self.nucleation_threshold

        # These 2D energy arrays account for most of the ``Ziptie's``
        # memory and processing cost.
        self.nucleation_energy = np.zeros((self.max_num_cables, 
                                           self.max_num_cables))
        self.agglomeration_energy = np.zeros((self.max_num_bundles, 
                                              self.max_num_cables))


    def learn(self, cable_surplus, bundle_activities):
        """ 
        Update co-activity estimates and calculate bundle activity 
        
        This step combines the projection of cables activities
        to bundle activities together with using the cable activities 
        to incrementally train the ``ZipTie``.

        Parameters
        ----------
        bundle_activities : array of floats
            Recent activity in each of the bundles..
        cable_surplus : array of floats
            The recent cable activities not accounted for in bundle activities.
            Row i corresponds to bundle i.

        Returns
        -------
        None
        """
        if not self.full:
            self._create_new_bundles(cable_surplus)
            self._grow_bundles(cable_surplus, bundle_activities)
        return

    #@jit(nopython=True)
    def _create_new_bundles(self, cable_surplus):
        """ 
        If the right conditions have been reached, create a new bundle.
        """
        # Incrementally accumulate nucleation energy.
        nucleation_energy_gather(cable_surplus, self.nucleation_energy)
   
        # Don't accumulate nucleation energy between a cable and itself.
        for i in range(cable_surplus.size):
            self.nucleation_energy[i,i] = 0.

        # Don't accumulate nucleation energy between cables already 
        # in the same bundle.
        for i_bundle in range(self.num_bundles):
            for ii_cable in range(self.num_members[i_bundle] - 1):
                for jj_cable in range(1, self.num_members[i_bundle]):
                    i_cable = self.members[i_bundle, ii_cable]
                    j_cable = self.members[i_bundle, jj_cable]
                    self.nucleation_energy[i_cable, j_cable] = 0.
                    self.nucleation_energy[j_cable, i_cable] = 0.
                         
        results  = -np.ones(3)
        max_dense(self.nucleation_energy, results)
        max_val = results[0]
        i_cable_a = int(results[1])
        i_cable_b = int(results[2])

        #print self.nucleation_energy
        # Add a new bundle if appropriate.
        if max_val > self.nucleation_threshold:
            self.members[self.num_bundles, 0] = i_cable_a 
            self.members[self.num_bundles, 1] = i_cable_b 
            self.num_members[self.num_bundles] = 2
            self.num_bundles += 1

            print ' '.join(['    ', self.name, 
                           'bundle', str(self.num_bundles), 
                           'added with cables', str(i_cable_a), 
                           str(i_cable_b)]) 

            # Check whether the ``ZipTie``'s capacity has been reached.
            if self.num_bundles == self.max_num_bundles:
                self.full = True

            # Reset the accumulated nucleation and agglomeration energy
            # for the two cables involved.
            self.nucleation_energy[i_cable_a, :] = 0.
            self.nucleation_energy[i_cable_b, :] = 0.
            self.nucleation_energy[:, i_cable_a] = 0.
            self.nucleation_energy[:, i_cable_b] = 0.
            self.agglomeration_energy[:, i_cable_a] = 0.
            self.agglomeration_energy[:, i_cable_b] = 0.

    def _grow_bundles(self, cable_surplus, bundle_activities):
        """ 
        Update an estimate of co-activity between all cables and all bundles.
        """
        # Incrementally accumulate agglomeration energy.
        agglomeration_energy_gather(bundle_activities, 
                                         cable_surplus,
                                         self.num_bundles,
                                         self.bundles_full,
                                         self.agglomeration_energy)

        # Don't accumulate agglomeration energy between cables already 
        # in the same bundle.
        for i_bundle in range(self.num_bundles):
            for ii_cable in range(self.num_members[i_bundle]):
                i_cable = self.members[i_bundle, ii_cable]
                self.agglomeration_energy[i_bundle, i_cable] = 0. 

        results  = -np.ones(3)
        max_dense(self.agglomeration_energy, results)
        max_energy = results[0]
        i_bundle = int(results[1])
        i_cable = int(results[2])

        # Add a new bundle if appropriate.
        if max_energy > self.agglomeration_threshold:

            # Check whether the agglomeration is already in the set of
            # created bundles. 
            # First find the indices of all the bundles that contain
            # all the cables of the proposed agglomeration.
            candidate_bundles = np.arange(self.num_bundles)
            bundle_matches = np.where(self.members == i_cable)[0]
            candidate_bundles = np.intersect1d(candidate_bundles, 
                                               bundle_matches)
            for i_cable_old in self.members[i_bundle, 
                                            :self.num_members[i_bundle]]:
                bundle_matches = np.where(self.members == i_cable_old)[0]
                candidate_bundles = np.intersect1d(candidate_bundles, 
                                                   bundle_matches)

            # Then check whether one of those bundles is the same size 
            # as the proposed agglomeration. Because the ziptie has
            # enforced uniqueness in each bundle's membership, this
            # shows whether there is an exact match to the proposed
            # agglomeration.
            match = False
            for i_candidate in candidate_bundles:
                if (self.num_members[i_candidate] == 
                    self.num_members[i_bundle] + 1):
                    match = True

            # If the agglomeration has already been used to create a 
            # bundle, ignore and reset they count. This can happen
            # under normal circumstances, because of how 
            # cable surpluses are calculated.
            if match:
                self.agglomeration_energy[i_bundle, i_cable] = 0.
                return

            # Add the agglomerated bundle to the membership map.
            # First copy the bundle it's being added to.
            for ii_cable in range(self.num_members[i_bundle]):
                self.members[self.num_bundles, ii_cable] = (
                        self.members[i_bundle, ii_cable] )

            # Add in the new cable. 
            self.num_members[self.num_bundles] = self.num_members[i_bundle]
            self.members[self.num_bundles, 
                         self.num_members[self.num_bundles]] = i_cable
            self.num_members[self.num_bundles] += 1

            # Check whether the bundle's capacity has been reached.
            if self.num_members[self.num_bundles] == self.max_cables_per_bundle:
                self.bundles_full[self.num_bundles] = True
            self.num_bundles += 1

            print ' '.join(['    ', self.name, 
                           'bundle', str(self.num_bundles), 
                           'added: bundle', str(i_bundle),
                           'and cable', str(i_cable)]) 

            # Check whether the ``ZipTie``'s capacity has been reached.
            if self.num_bundles == self.max_num_bundles:
                self.full = True

            # Reset the accumulated nucleation and agglomeration energy
            # for the two cables involved.
            self.nucleation_energy[i_cable, :] = 0.
            self.nucleation_energy[i_cable, :] = 0.
            self.nucleation_energy[:, i_cable] = 0.
            self.nucleation_energy[:, i_cable] = 0.
            self.agglomeration_energy[:, i_cable] = 0.
            self.agglomeration_energy[i_bundle, :] = 0.
    

    def get_elements(self, i_bundle):
        """
        Get the set of element indices that contribute to ith bundle.
        
        Parameters
        ----------
        i_bundle : int
            The index of the bundle for which to get element indices.

        Returns
        -------
        array of ints
            The indices of the elements that contribute to i_bundle.
        """
        return self.members[i_bundle, :self.num_members[i_bundle]]

    def get_index_projection(self, i_bundle):
        """ 
        Project ``i_bundle`` down to its cable indices.

        Parameters
        ----------
        i_bundle : int
            The index of the bundle to project onto its constituent cables.

        Returns
        -------
        projection : array of floats
            An array of zeros and ones, representing all the cables that
            contribute to the bundle. The values ``projection``
            corresponding to all the cables that contribute are 1.
        """
        projection = np.zeros(self.max_num_cables)
        for ii_cable in range(self.num_members[i_bundle]):
            projection[self.members[i_bundle,ii_cable]] = 1.
        return 
        

    def visualize(self):
        """
        Turn the state of the ``ZipTie`` into an image.
        """
        print self.name
        # First list the bundles and the cables in each.
        print 'Cables in each bundle'
        for i_bundle in range(self.num_bundles):
            print(' '.join([str(i_bundle), ':', 
                            str(self.members[i_bundle,
                                             :self.num_members[i_bundle]]) ]))

        plot = False
        if plot: 
            if self.num_bundles > 0:
                # Render the membership map.
                label = '_'.join([self.name, 'membership_map'])
                tools.visualize_array(self.members, label=label)

                # Render the agglomeration energy.
                label = '_'.join([self.name, 'agg_energy'])
                tools.visualize_array(self.agglomeration_energy, label=label)
                plt.xlabel( str(np.max(self.agglomeration_energy)) )

                # Render the nucleation energy.
                label = '_'.join([self.name, 'nuc_energy'])
                tools.visualize_array(self.nucleation_energy, label=label)
                plt.xlabel( str(np.max(self.nucleation_energy)) )


"""
Numba functions 
"""

@jit(nopython=True)
def max_dense(array2d, results):
    """
    Find the maximum value of a dense 2D array, with its row and column

    Parameters
    ----------
    array2d : 2D array of floats
        The array to find the maximum value of.
    results : array of floats
        This three element array gives the function a way to return results.
        results[0] : max_val
            The maximum value.
        results[1] : i_row_max 
        results[2] : i_col_max 
            The row and column containing the maximum value.

    Returns
    -------
    None. Results returned through modifying results.
    """
    max_val = results[0] 
    i_row_max = results[1] 
    i_row_min = results[2] 

    for i_row in range(array2d.shape[0]): 
        for i_col in range(array2d.shape[1]): 
            if array2d[i_row, i_col] > max_val:
                max_val = array2d[i_row, i_col]
                i_row_max = i_row
                i_col_max = i_col

    results[0] = max_val
    results[1] = i_row_max
    results[2] = i_col_max


@jit(nopython=True)
def nucleation_energy_gather(cable_surplus, nucleation_energy):
    """
    Gather nucleation energy.

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like 
        nucleation_energy += (cable_surplus * 
                              cable_surplus.T * 
                              nucleation_energy_rate) 
    Parameters
    ----------
    cable_surplus : array of floats
        The current activity of each input feature that is not explained by
        or captured in a bundle.
    nucleation_energy : 2D array of floats
        The amount of nucleation energy accumulated between each pair of
        input features.

    Results
    -------
    Returned indirectly by modifying ```nucleation_energy``.
    """
    for i_cable1 in range(len(cable_surplus)):
        activity1 = cable_surplus[i_cable1]
        if activity1 > 0.:
            for i_cable2 in range(len(cable_surplus)):
                activity2 = cable_surplus[i_cable2]
                if activity2 > 0.:
                    nucleation_energy[i_cable1, i_cable2] += (
                            activity1 * activity2) 

@jit(nopython=True)
def agglomeration_energy_gather(bundle_activities, cable_surplus,
                                n_bundles, bundles_full,
                                agglomeration_energy):
    """
    Accumulate the energy binding a new feature to an existing bundle..

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like 
        coactivities = bundle_activities * cable_surplus.T
        agglomeration_energy += coactivities * agglomeration_energy_rate

    Parameters
    ----------
    agglomeration_energy : 2D array of floats
        The total energy that has been accumulated between each cable
        and each bundle.
    bundle_activities : array of floats
        The activity level of each bundle.
    cable_surplus : array of floats
        The current activity of each cable that is not explained by
        or captured in a bundle.
    n_bundles : int
        The number of bundles that have been created so far.

    Results
    -------
    Returned indirectly by modifying ```agglomeration_energy``.
    """
    for i_col in range(len(cable_surplus)):
        activity = cable_surplus[i_col]
        if activity > 0.:
            # Only decay bundles that have been created
            for i_row in range(n_bundles):
                if bundle_activities[i_row] > 0.:
                    if not bundles_full[i_row]:
                        coactivity = activity * bundle_activities[i_row]
                        agglomeration_energy[i_row, i_col] += coactivity 

