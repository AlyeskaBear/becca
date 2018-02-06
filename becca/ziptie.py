"""
The Ziptie class.
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import becca.tools as tools
import becca.ziptie_numba as nb


class Ziptie(object):
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

    When stacked with other levels,
    zipties form a sparse deep neural network (DNN).
    This DNN has the extremely desirable characteristic of
    l-0 sparsity--the number of non-zero weights are minimized.
    The vast majority of weights in this network are zero,
    and the rest are one.
    This makes sparse computation feasible and allows for
    straightforward interpretation and visualization of the
    features.
    """

    def __init__(
            self,
            debug=False,
            n_cables=16,
            name=None,
            threshold=1e4,
    ):
        """
        Initialize the ziptie, pre-allocating data structures.

        Parameters
        ----------
        debug : boolean, optional
            Indicate whether to print informative status messages
            during execution. Default is False.
        n_cables : int
            The number of inputs to the Ziptie.
        name : str, optional
            The name assigned to the Ziptie.
            Default is 'ziptie'.
        threshold : float
            The point at which to nucleate a new bundle or
            agglomerate to an existing one.
        """
        if name is None:
            self.name = 'ziptie'
        else:
            self.name = name
        self.debug = debug
        self.n_cables = n_cables
        # n_bundles : int, optional
        #     The number of bundle outputs from the Ziptie.
        self.n_bundles = 0

        # nucleation_threshold : float
        #     Threshold above which nucleation energy results in nucleation.
        self.nucleation_threshold = threshold
        # agglomeration_threshold
        #     Threshold above which agglomeration energy results
        #     in agglomeration.
        self.agglomeration_threshold = self.nucleation_threshold
        # activity_threshold : float
        #     Threshold below which input activity is teated as zero.
        #     By ignoring the small activity values,
        #     computation gets much faster.
        self.activity_threshold = .1
        # cable_activities : array of floats
        #     The current set of input actvities.
        self.cable_activities = np.zeros(self.n_cables)
        # bundle_activities : array of floats
        #     The current set of bundle activities.
        self.bundle_activities = np.zeros(self.n_bundles)

        # bundle_to_cable_mapping: list of lists
        #     To get the cable indices for bundle i:
        #     [i_cables] = self.bundle_to_cable_mapping[i_bundle]
        #     An empty list shows an unused bundle.
        self.bundle_to_cable_mapping = [[]]
        # cable_to_bundle_mapping: list of lists
        #     To get the bundles that cable i contributes to:
        #     [i_bundles = self.cable_to_bundle_mapping[i_cable]
        #     An empty list shows that the cable is not a part
        #     of any bundle.
        self.cable_to_bundle_mapping = [[]] * self.n_cables

        # bundle_map_cols, bundle_map_rows : array of ints
        #     To represent the sparse 2D bundle map, a pair of row and col
        #     arrays are used. Rows are bundle indices, and cols are
        #     feature indices.  The bundle map shows which cables
        #     are zipped together to form which bundles.
        # self.bundle_map_rows = -np.ones(self.bundle_map_size).astype(int)
        # self.bundle_map_cols = -np.ones(self.bundle_map_size).astype(int)
        # n_map_entries: int
        #     The total number of bundle map entries that
        #     have been created so far.
        # self.n_map_entries = 0

        # agglomeration_energy: 2D array of floats
        #     The accumulated agglomeration energy for each
        #     bundle-cable pair. Bundles are represented in rows,
        #     cables are in columns.
        self.agglomeration_energy = np.zeros((self.n_cables,
                                              self.n_cables))
        # agglomeration_mask: 2D array of floats
        #     A binary array indicating which cable-bundle
        #     pairs are allowed to accumulate
        #     energy and which are not. Some combinations are
        #     disallowed because they result in redundant bundles.
        self.agglomeration_mask = np.ones((self.n_cables,
                                           self.n_cables))
        # nucleation_energy: 2D array of floats
        #     The accumualted nucleation energy associated
        #     with each cable-cable pair.
        self.nucleation_energy = np.zeros((self.n_cables,
                                           self.n_cables))
        # nucleation_mask: 2D array of floats
        #     A binary array indicating which cable-cable
        #     pairs are allowed to accumulate
        #     energy and which are not. Some combinations are
        #     disallowed because they result in redundant bundles.
        #     Make the diagonal zero to disallow cables to pair with
        #     themselves.
        self.nucleation_mask = (
            np.ones((self.n_cables, self.n_cables))
            - np.eye(self.n_cables))

    def update_bundles(self, new_cable_activities):
        """
        Calculate how much the cables' activities contribute to each bundle.

        Find bundle activities by taking the minimum input value
        in the set of cables in the bundle. The bulk of the computation
        occurs in ziptie_numba.find_bundle_activities.

        Parameters
        ----------
        new_cable_activities: array of floats

        Returns
        -------
        bundle_activities: array of floats
        """
        self.cable_activities = new_cable_activities
        self.bundle_activities = np.zeros(self.n_bundles)
        for i_bundle, i_cables in enumerate(self.bundle_to_cable_mapping):
            if len(i_cables) > 0:
                self.bundle_activities[i_bundle] = np.min(
                    self.cable_activities[i_cables])
        return self.bundle_activities

    def create_new_bundles(self):
        """
        If the right conditions have been reached, create a new bundle.
        """
        # Incrementally accumulate nucleation energy.
        nb.nucleation_energy_gather(self.cable_activities,
                                    self.nucleation_energy,
                                    self.nucleation_mask)
        max_energy, i_cable_a, i_cable_b = nb.max_2d(self.nucleation_energy)

        # Add a new bundle if appropriate
        if max_energy > self.nucleation_threshold:
            i_bundle = self.n_bundles
            self.increment_n_bundles()

            if len(self.bundle_to_cable_mapping) > i_bundle:
                self.bundle_to_cable_mapping[i_bundle] =[i_cable_a, i_cable_b]
            else:
                self.bundle_to_cable_mapping.append(
                    [i_cable_a, i_cable_b])
            self.cable_to_bundle_mapping[i_cable_a].append(i_bundle)
            self.cable_to_bundle_mapping[i_cable_b].append(i_bundle)

            # Reset the accumulated nucleation and agglomeration energy
            # for the two cables involved.
            self.nucleation_energy[i_cable_a, :] = 0
            self.nucleation_energy[i_cable_b, :] = 0
            self.nucleation_energy[:, i_cable_a] = 0
            self.nucleation_energy[:, i_cable_b] = 0
            self.agglomeration_energy[:, i_cable_a] = 0
            self.agglomeration_energy[:, i_cable_b] = 0

            # Update nucleation_mask to prevent the two cables from
            # accumulating nucleation energy in the future.
            self.nucleation_mask[i_cable_a, i_cable_b] = 0
            self.nucleation_mask[i_cable_b, i_cable_a] = 0

            # Update agglomeration_mask to account for the new bundle.
            # The new bundle should not accumulate agglomeration energy
            # with any of the cables that any of its constituent cables
            # are blocked from nucleating with.
            blocked_a = np.where(self.nucleation_mask[i_cable_a, :] == 0)[0]
            blocked_b = np.where(self.nucleation_mask[i_cable_b, :] == 0)[0]
            blocked = np.union1d(blocked_a, blocked_b)
            self.agglomeration_mask[i_bundle, blocked] = 0


            if self.debug:
                print(' '.join([
                    '    ', self.name,
                    'bundle', str(i_bundle),
                    'added with cables', str(i_cable_a),
                    str(i_cable_b)
                ]))

    def grow_bundles(self):
        """
        Update an estimate of co-activity between all cables.
        """
        # Incrementally accumulate agglomeration energy.
        nb.agglomeration_energy_gather(self.bundle_activities,
                                       self.cable_activities,
                                       self.n_bundles,
                                       self.agglomeration_energy,
                                       self.agglomeration_mask)
        max_energy, i_bundle, i_cable = nb.max_2d(self.agglomeration_energy)

        # Add a new bundle if appropriate
        if max_energy > self.agglomeration_threshold:
            # Add the new bundle to the end of the list.
            i_new_bundle = self.n_bundles
            self.increment_n_bundles()

            # Make a copy of the growing bundle.
            self.bundle_to_cable_mapping.append(
                self.bundle_to_cable_mapping[i_bundle])
            # Add in the new cable.
            self.bundle_to_cable_mapping[i_new_bundle].append(i_cable)
            # Update the contributing cables.
            for i_cable in self.bundle_to_cable_mapping[i_new_bundle]:
                self.cable_to_bundle_mapping[i_cable].append(i_new_bundle)

            # Reset the accumulated nucleation and agglomeration energy
            # for the two cables involved.
            self.nucleation_energy[i_cable, :] = 0.
            self.nucleation_energy[i_cable, :] = 0.
            self.nucleation_energy[:, i_cable] = 0.
            self.nucleation_energy[:, i_cable] = 0.
            self.agglomeration_energy[:, i_cable] = 0.
            self.agglomeration_energy[i_bundle, :] = 0.

            # Update agglomeration_mask to account for the new bundle.
            # The new bundle should not accumulate agglomeration energy with
            # 1) the cables that its constituent cable
            #    are blocked from nucleating with or
            # 2) the cables that its constituent bundle
            #    are blocked from agglomerating with.
            blocked_cable = np.where(
                self.nucleation_mask[i_cable, :] == 0.)
            blocked_bundle = np.where(
                self.agglomeration_mask[i_bundle, :] == 0.)
            blocked = np.union1d(blocked_cable[0], blocked_bundle[0])
            self.agglomeration_mask[i_new_bundle, blocked] = 0.

            if self.debug:
                print(' '.join(['    ', self.name,
                                'bundle', str(i_new_bundle),
                                'added: bundle', str(i_bundle),
                                'and cable', str(i_cable)]))

    def update_inputs(self, resets):
        """
        Reset indicated cables and all the bundles associated with them.

        Parameters
        ----------
        resets: array of ints
            The indices of the cables that are being reset

        Returns
        -------
        upstream_resets: array of ints
            The indices of the bundles to be reset.
        """

        upstream_resets = []
        for i_cable in resets:
            for i_bundle in self.cable_to_bundle_mapping[i_cable]:
                upstream_resets.append(i_bundle)
                # Remove the bundle from the mappings in both directions.
                for j_cable in self.bundle_to_cable_mapping[i_bundle]:
                    self.cable_to_bundle_mapping[j_cable].remove(i_bundle)
                self.bundle_to_cable_mapping[i_bundle] = []

                self.agglomeration_mask[i_bundle, :] = 1
                self.agglomeration_energy[i_bundle, :] = 0

            self.agglomeration_mask[:, i_cable] = 1
            self.agglomeration_energy[:, i_cable] = 0

            self.nucleation_mask[i_cable, :] = 1
            self.nucleation_mask[:, i_cable] = 1
            self.nucleation_mask[i_cable, i_cable] = 0
            self.nucleation_energy[i_cable, :] = 0
            self.nucleation_energy[:, i_cable] = 0

        return upstream_resets

    def increment_n_bundles(self):
        """
        Add one to n_map entries and grow the bundle map as needed.
        """
        self.n_bundles += 1
        n_max_bundles = self.agglomeration_energy.shape[0]
        if self.n_bundles >= n_max_bundles:
            new_max_bundles = n_max_bundles * 2
            new_agglomeration_energy = np.zeros(
                (new_max_bundles, self.n_cables))
            new_agglomeration_energy[:new_max_bundles, :] = (
                    self.agglomeration_energy)
            self.agglomeration_energy = new_agglomeration_energy

            new_agglomeration_mask = np.zeros(
                (new_max_bundles, self.n_cables))
            new_agglomeration_mask[:new_max_bundles, :] = (
                    self.agglomeration_mask)
            self.agglomeration_mask = new_agglomeration_mask

    def get_index_projection(self, i_bundle):
        """
        Project i_bundle down to its cable indices.

        Parameters
        ----------
        i_bundle : int
            The index of the bundle to project onto its constituent cables.

        Returns
        -------
        projection : array of floats
            An array of zeros and ones, representing all the cables that
            contribute to the bundle. The values projection
            corresponding to all the cables that contribute are 1.
        """
        projection = np.zeros(self.n_cables)
        projection[self.get_index_projection_cables()] = 1
        return projection

    def get_index_projection_cables(self, i_bundle):
        """
        Project i_bundle down to its cable indices.

        Parameters
        ----------
        i_bundle : int
            The index of the bundle to project onto its constituent cables.

        Returns
        -------
        projection_indices : array of ints
            An array of cable indices, representing all the cables that
            contribute to the bundle.
        """
        projection_indices = self.bundle_to_cable_mapping[i_bundle]
        return projection_indices


    def project_bundle_activities(self, bundle_activities):
        """
        Take a set of bundle activities and project them to cable activities.

        Parameters
        ----------
        bundle_activities: array of floats

        Results
        -------
        cable_activities: array of floats
        """
        cable_activities = np.zeros(self.n_cables)
        for i_cable, i_bundles in enumerate(self.cable_to_bundle_mapping):
            if len(i_bundles) > 0:
                cable_activities[i_cable] = np.min(
                    self.bundle_activities[i_bundles])
        return cable_activities

    def visualize(self):
        """
        Turn the state of the Ziptie into an image.
        """
        print(self.name)
        # First list the bundles and the cables in each.
        i_bundles = self.bundle_map_rows[:self.n_map_entries]
        i_cables = self.bundle_map_cols[:self.n_map_entries]
        i_bundles_unique = np.unique(i_bundles)
        if i_bundles_unique is not None:
            for i_bundle in i_bundles_unique:
                b_cables = list(np.sort(i_cables[np.where(
                    i_bundles == i_bundle)[0]]))
                print(' '.join(['    bundle', str(i_bundle),
                                'cables:', str(b_cables)]))

        plot = False
        if plot:
            if self.n_map_entries > 0:
                # Render the bundle map.
                bundle_map = np.zeros((self.n_cables,
                                       self.n_bundles))
                nb.set_dense_val(bundle_map,
                                 self.bundle_map_rows[:self.n_map_entries],
                                 self.bundle_map_cols[:self.n_map_entries], 1.)
                tools.visualize_array(bundle_map,
                                      label=self.name + '_bundle_map')

                # Render the agglomeration energy.
                label = '_'.join([self.name, 'agg_energy'])
                tools.visualize_array(self.agglomeration_energy, label=label)
                plt.xlabel(str(np.max(self.agglomeration_energy)))

                # Render the nucleation energy.
                label = '_'.join([self.name, 'nuc_energy'])
                tools.visualize_array(self.nucleation_energy, label=label)
                plt.xlabel(str(np.max(self.nucleation_energy)))
