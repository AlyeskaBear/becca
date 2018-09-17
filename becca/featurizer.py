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
    ):
        """
        Configure the featurizer.

        Parameters
        ---------
        debug: boolean
        n_inputs : int
            The number of inputs (cables) that each Ziptie will be
            equipped to handle.
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
            n_inputs=self.n_inputs,
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
            debug=self.debug)

        # n_features: int
        #     The total number of features that have been colllected so far.
        #     This includes the cable candidate pools from each ziptie.
        self.n_features_by_level = [0, 0]

        # mapping: 2D array of ints
        #     The transformation from candidates (List or arrays of values)
        #     to the feature pool.
        #     If there is a one at [row_i, col_j] then
        #     candidate row_i maps to feature index col_j .
        #         feature_pool = np.matmul(candidates, self.mapping)
        self.mapping = np.zeros((0, 0), dtype=np.int)

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

        self.activities = [
            self.ziptie_0_cable_pool,
            ziptie_1_cable_pool,
        ]

        self.feature_pool = self.map_to_feature_pool(self.activities)

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
        pool_fitness = self.filter.update_fitness(cable_fitness)

        return pool_fitness

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
            i_start = np.sum(np.array(self.n_features_by_level[:i_level]))
            for i_reset in level_resets:
                resets.append(np.where(
                    self.mapping[i_start + i_reset, :])[0])

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
        candidate_values = []
        i_last = 0
        for n_feat in self.n_features_by_level:
            candidate_values.append(
                feature_values[i_last: i_last + n_feat])
            i_last += n_feat
        return candidate_values

    def map_to_feature_pool(self, candidate_values):
        """
        Map the candidates over all levels to the appropriate
        feature candidates.

        Parameters
        ----------
        candidate_values: list of arrays of floats

        Returns
        -------
        feature_values: array of floats
        """
        self.grow_map(candidate_values)
        all_candidate_values = []
        for level_candidate_values in candidate_values:
            all_candidate_values += list(level_candidate_values)
        feature_values = np.matmul(
            np.array(all_candidate_values), self.mapping)
        return feature_values

    def grow_map(self, candidate_values):
        """
        Check whether we need to add more candidates to the feature pool.

        New candidates will come in appended to the end of their
        respective input pools. However, feature indices need to
        stay consistent throughout the life of each feature.
        these new candidates need to be given indices at the
        end of the currently used set of feature pool indices.

        Parameters
        ----------
        candidate_values: list of arrays of floats
        """
        # Check whether the number of candidates has expanded
        # at any level and adapt.
        n_candidates_by_level = [
            values.size for values in candidate_values]
        total_n_candidates = np.sum(np.array(n_candidates_by_level))
        total_n_features = np.sum(np.array(self.n_features_by_level))

        if (total_n_features < total_n_candidates):
            # Create a larger map
            new_mapping = []
            i_last_old = 0  # Track last candidate handled.
            j_last_new = total_n_features  # Track last feature assigned.

            for i_level in range(len(self.n_features_by_level)):
                n_cand = n_candidates_by_level[i_level]
                n_feat = self.n_features_by_level[i_level]
                delta = n_cand - n_feat

                level_old_map = np.zeros((n_feat, total_n_candidates))
                level_old_map[:, :total_n_features] = self.mapping[
                    i_last_old:i_last_old + n_feat, :]
                new_mapping.append(level_old_map)

                if delta > 0:
                    level_new_map = np.zeros((delta, total_n_candidates))
                    level_new_map[
                        :,
                        j_last_new: j_last_new + delta
                    ] = np.eye(delta)
                    new_mapping.append(level_new_map)

                    j_last_new += delta
                    self.n_features_by_level[i_level] += delta

                i_last_old += n_feat
            self.mapping = np.concatenate(new_mapping)
        return
