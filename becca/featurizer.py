import numpy as np

from becca.input_filter import InputFilter
from becca.ziptie import Ziptie


class Featurizer(object):
    """
    Convert inputs to bundles and learn new bundles.
    Inputs are transformed into bundles, sets of inputs
    that tend to co-occur.
    """
    def __init__(
        self,
        debug=False,
        n_inputs=None,
        threshold=None,
    ):
        """
        Configure the featurizer.

        Parameters
        ---------
        debug: boolean
        n_inputs : int
            The number of inputs (cables) that each Ziptie will be
            equipped to handle.
        threshold : float
            See Ziptie.nucleation_threshold
        """
        self.debug = debug

        # name: string
        #     A label for this object.
        self.name = 'featurizer'

        # epsilon: float
        #     A constant small theshold used to test for significant
        #     non-zero-ness.
        self.epsilon = 1e-8

        # n_inputs: int
        #     The maximum numbers of inputs and bundles
        #     that this level can accept.
        self.n_inputs = n_inputs

        # TODO:
        # Move input filter and ziptie creation into step,
        # along with clustering.

        # filter: InputFilter
        #     Reduce the possibly large number of inputs to the number
        #     of cables that the Ziptie can handle. Each Ziptie will
        #     have its own InputFilter.
        self.filter = InputFilter(
            n_inputs = self.n_inputs,
            name='ziptie_0',
            debug=self.debug,
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
            threshold=threshold,
            debug=self.debug)

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
        all_input_fitness = self.map_from_feature_pool(feature_fitness)

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
        bundle_resets = self.ziptie.update_inputs(filter_resets)
        # Leave an empty list of resets for lowest level input.
        # They are always all passed in as feature candidates.
        # They never get reset or swapped out. The model's input filter
        # deals with them.
        # As other levels are created, append their bundle resets as well.
        all_resets = [[], bundle_resets]

        resets = []
        for i_level, level_resets in enumerate(all_resets):
            for i_reset in level_resets:
                resets.append(self.mapping_to_features[i_level][i_reset])

        return resets

    def map_from_feature_pool(self, feature_values):
        """
        For an array corresponding to feature candidates from the model,
        generate and order those values corresponding to ziptie candidates.

        Parameters
        ----------
        feature_values: array of floats

        Returns
        -------
        candidate_values: list of array of floats
        """
        candidate_values = [np.zeros(self.n_inputs)]
        candidate_values.append(np.zeros(self.ziptie.n_bundles))
         
        for i_feature, (i_level, i_candidate) in enumerate(
                self.mapping_from_features):
            candidate_values[i_level][i_candidate] = feature_values[i_feature]
        return candidate_values

    def map_to_feature_pool(self, candidate_values):
        """
        Map the candidates over all levels to the appropriate feature candidates.

        Parameters
        ----------
        candidate_values: list of arrays of floats

        Returns
        -------
        feature_values: array of floats
        """
        # Check whether the number of candidates has expanded at any level
        # and adapt.
        for _ in range(len(self.mapping_to_features), len(candidate_values)):
            self.mapping_to_features.append([])
        for i_level, ziptie_cable_pool in enumerate(candidate_values):
            # Map any unmapped candidates to features.
            for i_new_candidate in range(len(self.mapping_to_features[i_level]),
                                     ziptie_cable_pool.size):
                self.mapping_to_features[i_level].append(
                    len(self.mapping_from_features))
                self.mapping_from_features.append((i_level, i_new_candidate))

        feature_values = np.zeros(len(self.mapping_from_features))
        for i_level, level_mapping in enumerate(self.mapping_to_features):
            for i_candidate, i_feature in enumerate(level_mapping):
                feature_values[i_feature] = candidate_values[i_level][i_candidate]

        return feature_values

    def featurize(self, new_candidates):
        """
        Learn bundles and calculate bundle activities.

        Parameters
        ----------
        new_candidates : array of floats
            The candidates collected by the brain for the current time step.

        Returns
        -------
        feature_pool: array of floats
        """
        self.ziptie_0_cable_pool = new_candidates
        cable_activities = self.filter.update_activities(
            candidate_activities=self.ziptie_0_cable_pool)

        # Incrementally update the bundles in the ziptie.
        self.ziptie.create_new_bundles()
        self.ziptie.grow_bundles()

        # Run the inputs through the ziptie to find bundle activities
        # and to learn how to bundle them.
        ziptie_1_cable_pool = self.ziptie.update_bundles(cable_activities)

        all_ziptie_cable_pool = [
            self.ziptie_0_cable_pool,
            ziptie_1_cable_pool,
        ]

        self.feature_pool = self.map_to_feature_pool(all_ziptie_cable_pool)

        return self.feature_pool

    def defeaturize(self, feature_pool):
        """
        Take a set of feature activities and represent them in candidates.
        """
        ziptie_0_cable_pool, ziptie_1_cable_pool = (
            self.map_from_feature_pool(feature_pool))
        # TODO: iterate over multiple zipties
        ziptie_0_cables = self.ziptie.project_bundle_activities(
            ziptie_1_cable_pool)
        ziptie_0_cable_pool_upstream = self.filter.project_activities(
            ziptie_0_cables)
        n_candidates_0 = ziptie_0_cable_pool_upstream.size
        ziptie_0_cable_pool = np.maximum(
            ziptie_0_cable_pool[:n_candidates_0],
            ziptie_0_cable_pool_upstream)
        return ziptie_0_cable_pool
