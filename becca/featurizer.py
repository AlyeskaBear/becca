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
            # max_n_features=None,
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
        # max_n_features : int
        #     The maximum numbers of inputs, bundles and features
        #     that this level can accept.
        #     max_n_features = n_inputs + n_bundles
        self.n_inputs = n_inputs
        # self.max_n_features = self.n_inputs
        self.n_bundles = 4 * self.n_inputs
        # if max_n_features is None:
        #     # Choose the total number of bundles (created features) allowed,
        #     # in terms of the number of inputs allowed.
        #     self.n_bundles = 3 * self.n_inputs
        #     self.max_n_features = self.n_inputs + self.n_bundles
        # else:
        #     self.max_n_features = max_n_features
        #     self.n_bundles = self.max_n_features - self.n_inputs

        # Normalization constants.
        # input_max : array of floats
        #     The maximum of each input's activity.
        #     Start with the assumption that each input has a
        #     maximum value of zero. Then adapt up from there
        #     based on the incoming observations.
        # input_max_decay_time, input_max_grow_time : float
        #     The time constant over which maximum estimates are
        #     decreased/increased. Growing time being lower allows the
        #     estimate to be weighted toward the maximum value.
        #self.input_max = np.zeros(self.n_inputs)
        #self.input_max_grow_time = 1e2
        #self.input_max_decay_time = self.input_max_grow_time * 1e2

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

    def calculate_fitness():
        pass

    def update_inputs():
        pass

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
        cable_activities, cable_resets, cable_fitness = self.filter.step(
            candidate_activities=self.input_activities, 

        # Run the inputs through the ziptie to find bundle activities
        # and to learn how to bundle them.
        bundle_activities = self.ziptie.featurize(self.input_activities)
        # The element activities are the combination of the residual
        # input activities and the bundle activities.
        self.feature_activities = np.concatenate((self.input_activities,
                                                  bundle_activities))

        # Track features that are active.
        # self.live_features[np.where(
        #     self.feature_activities > self.epsilon)] = 1.

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
