"""
The Featurizer class.
"""

from __future__ import print_function
import numpy as np

import becca.featurizer_viz as viz
from becca.ziptie import Ziptie


class Featurizer(object):
    """
    Convert inputs to bundles and learn new bundles.

    Inputs are transformed into bundles, sets of inputs that tend to co-occur.
    """
    def __init__(
            self,
            n_inputs,
            threshold=None,
            verbose=False,
            ):
        """
        Configure the featurizer.

        Parameters
        ---------
        n_inputs : int
            The number of inputs (cables) that each Ziptie will be
            equipped to handle.
        threshold : float
            See Ziptie.nucleation_threshold
        """
        # verbose : boolean
        #     Print out extra information about the featurizer's operation.
        self.verbose = verbose

        # name : string
        #     A label for this object.
        self.name = 'featurizer'

        # epsilon : float
        #     A constant small theshold used to test for significant
        #     non-zero-ness.
        self.epsilon = 1e-8

        # n_inputs : int
        # n_bundles : int
        #     The maximum numbers of inputs and bundles
        #     that this level can accept.
        self.n_inputs = n_inputs
        self.n_bundles = 4 * self.n_inputs

        # input_activities,
        # bundle_activities,
        # feature_activities : array of floats
        #     inputs, bundles and features are characterized by their
        #     activity--their level of activation at each time step.
        #     Activity for each input or bundle or feature
        #     can vary between zero and one.
        #self.input_activities = np.zeros(self.n_inputs)
        #self.bundle_activities = np.zeros(self.n_bundles)
        #self.feature_activities = np.zeros(self.max_n_features)

        # live_features : array of floats
        #     A binary array tracking which of the features have ever
        #     been active.
        # self.live_features = np.zeros(self.max_n_features)

        # TODO:
        # Move input filter and ziptie creation into step,
        # along with clustering.

        # filter: InputFilter
        #     Reduce the possibly large number of inputs to the number
        #     of cables that the Ziptie can handle. Each Ziptie will
        #     have its own InputFilter.
        self.filter = InputFilter(
            n_inputs_final = self.n_inputs,
            verbose=self.verbose,
        )
        # ziptie: Ziptie
        #     The ziptie is an instance of the Ziptie algorithm class,
        #     an incremental method for bundling inputs. Check out
        #     ziptie.py for a complete description. Zipties note which
        #     inputs tend to be co-active and creates bundles of them.
        #     This feature creation mechanism results in l0-sparse
        #     features, which sparsity helps keep Becca fast.
        self.ziptie = Ziptie(
            n_cables=self.n_inputs,
            n_bundles=self.n_bundles,
            threshold=threshold,
            verbose=self.verbose)

        # mapping_to_features: list of lists
        #     Tracks which feature candidate index corresponds to each
        #     ziptie input. The first level list corresponds to
        #         0: inputs to ziptie level 0
        #         1: inputs to ziptie level 1
        #               (also bundles from ziptie level 0)
        #         2: inputs to ziptie level 2
        #               (also bundles from ziptie level 1)
        #         ...
        #     The second level list index of the feature candidate that
        #     each input maps to.
        self.mapping_to_features = [[]]
        # mapping_from_features: list of 2-tuples,
        #     The inverse of mapping_to_features. Tracks which input
        #    corresponds which each feature candidate.
        #     Each item is of the format
        #         (level, input index)
        self.mapping_from_features = []
        # If mapping_from_features[i] = (j, k)
        # then mapping_to_features[j, k] = i

    def calculate_fitness(self, feature_fitness):
        """
        Find the predictive fitness of each of cables in each ziptie.

        Parameters
        ----------
        candidate_fitness: array of floats
        """
        # TODO: Handle a hierarchy of zipties and input filters
        # Include upstream fitness
        all_input_fitness = self.map_features_to_inputs(feature_fitness)

        cable_fitness = self.ziptie.project_bundle_activities(
            all_input_fitness[1])
        input_fitness = np.maximum(cable_fitness, all_input_fitness[0])
        self.filter.update_fitness(input_fitness)

    def update_inputs(self):
        """
        Give each input filter a chance to update their inputs and propagate
        any resets that result up through the zipties and to the model.

        Returns
        -------
        resets: array of ints
            The feature candidate indices that are being reset. 
        """
        filter_resets = self.filter.update_inputs()
        bundle_resets = self.ziptie.update_inputs()
        all_resets = [filter_resets, bundle_resets]

        resets = []
        for i_level, level_resets in enumerate(all_resets):
            for i_reset in level_resets:
                resets.append(self.mapping_to_features[i_level][i_reset])

        return resets

    def map_features_to_inputs(self, feature_values):
        """
        For an array corresponding to feature candidates from the model,
        generate and order those values corresponding to ziptie inputs.

        Parameters
        ----------
        feature_values: array of floats

        Returns
        -------
        input_values: list of array of floats
        """
        input_values = [np.zeros(self.input_activities.size)]
        input_values.append(np.zeros(self.ziptie.n_bundles))
        
        for i_feature, (i_level, i_input) in enumerate(
                self.mapping_from_features):
            input_values[i_level][i_input] = feature_values[i_feature]
        return input_values

    def map_inputs_to_features(self, input_values):
        """
        Map the inputs over all levels to the appropriate feature candidates.

        Parameters
        ----------
        input_values: list of arrays of floats

        Returns
        -------
        feature_values: array of floats
        """
        feature_values = np.zeros(???)
        for i_level, level_mapping in enumerate(self.mapping_to_features):
            for i_input, i_feature in enumerate(level_mapping):
                feature_values[i_feature] = input_values[i_level][i_input]

        return feature_values

    def featurize(self, new_inputs):
        """
        Learn bundles and calculate bundle activities.

        Parameters
        ----------
        new_inputs : array of floats
            The inputs collected by the brain for the current time step.

        Returns
        -------
        feature_activities: array of floats
        """
        # Start by normalizing all the inputs.
        #self.input_activities = self.update_inputs(new_inputs)
        self.input_activities = new_inputs
        cable_activities = self.filter.update_activities(
            candidate_activities=self.input_activities)

        # Run the inputs through the ziptie to find bundle activities
        # and to learn how to bundle them.
        bundle_activities = self.ziptie.featurize(self.input_activities)

        all_input_activities = [
            self.input_activities,
            bundle_activities,
        ]

        # Check whether the number of inputs has expanded at any level
        # and adapt.
        for i_level, input_activities in enumerate(all_input_activities):
            # Map any unmapped inputs to features.
            for i_new_input in range(len(self.mapping_to_features[i_level]),
                                     input_activities.size):
                i_feature = len(self.mapping_from_features)
                self.mapping_to_features[i_level].append(i_feature)
                self.mapping_from_features.append((i_level, i_new_input))

        # The element activities are the combination of the residual
        # input activities and the bundle activities
        self.feature_activities = self.map_inputs_to_features([

        # Incrementally update the bundles in the ziptie.
        self.ziptie.learn(self.input_activities)

        return self.feature_activities


    def defeaturize(self, feature_activities):
        """
        Take a set of feature activities and represent them in inputs.
        """
        input_activities = feature_activities[:self.n_inputs]
        # Project each ziptie down to inputs.
        bundle_activities = feature_activities[self.n_inputs:]
        # TODO: iterate over multiple zipties
        ziptie_input_activities = self.ziptie.project_bundle_activities(
            bundle_activities)
        input_activities = np.maximum(
            input_activities, ziptie_input_activities)
        return input_activities


    def update_fitness(self, feature_fitness):
        """
        Recalculate the fitness of each cable candidate in each ziptie.

        Parameters
        ----------
        feature_fitness: array of floats
        """
        # TODO: Cable candidate fitness is a combination of feature
        # fitness (from the model) and the feature fitness of any bundle
        # with which
        # the cable candidate might be affiliated.
        pass


    # TODO: Remove ziptie masks and update_masks()
    def update_masks(self, new_input_indices):
        """
        Upate the energy masks in the ziptie.

        @param new_input_indices: list of tuples of (int, int)
           Tuples of (child_index, parent_index). Each time a new child
           node is added, it is recorded on this list.
        """
        for pair in new_input_indices:
            self.ziptie.update_masks(pair[0], pair[1])

    '''
    def update_inputs(self, inputs):
        """
        Normalize and update inputs.

        Normalize activities so that they are predictably distrbuted.
        Use a running estimate of the maximum of each cable activity.
        Scale it so that the max would fall at 1.

        Normalization has several benefits.
        1. It makes for fewer constraints on worlds and sensors.
           It allows any sensor can return any range of values.
        2. Gradual changes in sensors and the world can be adapted to.
        3. It makes the bundle creation heuristic more robust and
           predictable. The approximate distribution of cable
           activities is known and can be designed for.

        After normalization, update each input activity with either
        a) the normalized value of its corresponding input or
        b) the decayed value, carried over from the previous time step,
        whichever is greater.

        Parameters
        ----------
        inputs : array of floats
            The current and previous activity of the inputs.

        Returns
        -------
        None
            self.input_activities is modified to include
            the normalized values of each of the inputs.
        """
        # TODO: numpy-ify this

        if inputs.size > self.n_inputs:
            print("Featurizer.update_inputs:")
            print("    Attempting to update out of range input activities.")

        # This is written to be easily compilable by numba, however,
        # it hasn't proven to be at all significant in profiling, so it
        # has stayed in slow-loop python for now.
        stop_index = min(inputs.size, self.n_inputs)
        # Input index
        j = 0
        for i in range(stop_index):
            val = inputs[j]

            # Decay the maximum value.
            self.input_max[i] += ((val - self.input_max[i]) /
                                  self.input_max_decay_time)

            # Eventually move this pre-processing to brain.py?
            # Grow the maximum value, when appropriate.
            if val > self.input_max[i]:
                self.input_max[i] += ((val - self.input_max[i]) /
                                      self.input_max_grow_time)

            # Scale the input by the maximum.
            val = val / (self.input_max[i] + self.epsilon)
            # Ensure that 0 <= val <= 1.
            val = max(0., val)
            val = min(1., val)

            self.input_activities[i] = val
            j += 1
        return self.input_activities
    '''


    def visualize(self, brain, world=None):
        """
        Show the current state of the featurizer.
        """
        viz.visualize(self, brain, world)
