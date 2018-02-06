from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import becca.tools as tools


class InputFilter(object):
    """
    The InputFilter selects a few inputs from among many candidates
    in the input pool.
    
    The selection process is driven by candidates' fitness and by how
    much activity they have shown.
    Each ziptie will have one, as will the model.
    """
    def __init__(self, debug=False, n_inputs=None, name='filter'):
        """
        Parameters
        ----------
        debug: boolean
        n_inputs: int
            The number of inputs that the filter is expected to maintain.
        name: string
            A string that helps to identify this input filter uniquely.
        """
        self.name = name
        # Check for valid arguments.
        if not n_inputs:
            print('You have to give a number for n_inputs.')
            return
        else:
            self.n_inputs = n_inputs

        # n_candidates: int
        #     The current number of candidates.
        self.n_candidates = 0
        
        # input_mapping: array of ints
        #     A mapping from candidates to inputs.
        #     This should be at least as large as the number of candidates.
        #     Position in this array shows which candidate is being assigned.
        #     The value at that position shows the input index it is
        #     assigned to. An index of -1 means that\
        #     candidate is unassigned.
        self.input_mapping = -np.ones(self.n_inputs * 2, dtype='int')

        # candidate_fitness: array of floats
        #     The most recently observed predictive fitness
        #     of each candidate.
        self.candidate_fitness = np.zeros(self.n_inputs * 2)

        # cumulative_activities: array of floats
        #     The accumulated activity over the lifetime of the candidate.
        self.cumulative_activities = np.zeros(self.n_inputs * 2)

        # bench_pressure: array of floats
        #     The 'impatience' of the candidate at not being used as
        #     an input. It is related to how long the candidate has
        #     been sitting unused.
        self.bench_pressure = np.zeros(self.n_inputs * 2)

        # pressure_time: float
        #     A time constant roughly determining the time scale over
        #     which a candidate's bench pressure will drive it into
        #     the input pool.
        self.pressure_time = 1e5

    def update_activities(self, candidate_activities):
        """
        Generate a new set of input activities.

        Parameters
        ----------
        candidate_activities: array of floats

        Returns
        -------
        input_activities: array of floats
        """
        # Grow the candidate-related attributes if necessary.
        self.n_candidates = candidate_activities.size
        capacity = self.input_mapping.size
        if self.n_candidates >= capacity:
            new_input_mapping = -np.ones(self.n_candidates * 2, dtype='int')
            new_input_mapping[:capacity] = self.input_mapping
            self.input_mapping = new_input_mapping

            new_candidate_fitness = np.zeros(self.n_candidates * 2)
            new_candidate_fitness[:capacity] = self.candidate_fitness
            self.candidate_fitness = new_candidate_fitness

            new_cumulative_activities = np.zeros(self.n_candidates * 2)
            new_cumulative_activities[:capacity] = (
                self.cumulative_activities)
            self.cumulative_activities = new_cumulative_activities

            new_bench_pressure = np.zeros(self.n_candidates * 2)
            new_bench_pressure[:capacity] = self.bench_pressure
            self.bench_pressure = new_bench_pressure


        input_activities = np.zeros(self.n_inputs)
        # This for loop is slow, but it's clear. It doesn't cost much.
        for i, loc in enumerate(self.input_mapping):
            if loc >= 0:
                input_activities[loc] = candidate_activities[i]        

        self.cumulative_activities[:self.n_candidates] += candidate_activities

        
        self.i_in_use = np.where(self.input_mapping >= 0)[0]
        i_unassigned = np.where(self.input_mapping == -1)[0]
        self.i_benched = i_unassigned[np.where(
            i_unassigned < self.n_candidates)]
        self.bench_pressure[self.i_benched] += (
            candidate_activities[self.i_benched] / (tools.epsilon + 
            self.cumulative_activities[self.i_benched] * self.pressure_time))

        return input_activities

    def project_activities(self, input_activities):
        """
        Project a set of input activities to their respective candidates.

        Parameters
        ----------
        input_activities: array of floats

        Returns
        -------
        candidate_activities: array of floats
        """
        candidate_activities = np.zeros(self.n_candidates)
        for i, loc in enumerate(self.input_mapping):
            if loc >= 0:
                candidate_activities[i] = input_activities[loc]
        return candidate_activities

    def update_fitness(self, feature_fitness):

        """
        Substitute the latest feature fitness values into candidate fitness.

        Parameters
        ----------
        feature_fitness: array of floats
            The predictive fitness of the current set of features.

        Returns
        -------
        candidate_fitness: array of floats
            The most recently observed predictive fitness of each candidate.
        """
        # This for loop is slow, but it's clear. It doesn't cost much.
        for i, loc in enumerate(self.input_mapping):
            if i >= 0:
                self.candidate_fitness[i] = feature_fitness[loc]        
        return self.candidate_fitness

    def update_inputs(self, upstream_resets=[]):
        """
        Re-evaluate which candidates should be inputs. Modify input mapping
        to add new candidates swap out underperforming ones.
        Issue resets wherever reassignments occur.

        Parameters
        ----------
        upstream_resets: list of ints
            Indices of the candidates to remove from the input filter.
            If any of these are in the outgoing set of inputs, send a resest
            so that their influence can be wiped clean.

        Returns
        -------
        resets: array of ints
            The indices of the inputs which need to be reset.
        """
        # Before doing anything else, handle upstream resets.
        for i_reset in upstream_resets:
            self.candidate_fitness[i_reset] = 0.
            self.bench_pressure[i_reset] = 0.
            self.cumulative_activities[i_reset] = 0.

        resets = []
        candidate_score = self.candidate_fitness + self.bench_pressure
        # Find lowest scoring candidates in use.
        i_lowest_scoring_in_use = np.argsort(
            candidate_score[self.i_in_use])[::-1]
        # Find highest scoring benched candidates.
        i_highest_scoring_benched = np.argsort(
            candidate_score[self.i_benched])

        # First fill out any unused inputs with candidates.
        n_inputs_used = np.max(self.input_mapping) + 1
        n_inputs_unassigned = self.n_inputs - n_inputs_used
        i_fill = 0
        while(n_inputs_unassigned > 0 and
              i_highest_scoring_benched.size > i_fill):
            i_in = self.i_benched[i_highest_scoring_benched[i_fill]]
            self.input_mapping[i_in] = self.n_inputs - n_inputs_unassigned
            # No need to specify resets.
            # There's no previous activity to clear.
            n_inputs_unassigned -= 1
            i_fill += 1

        i_highest_scoring_benched = i_highest_scoring_benched[i_fill:]

        # Then swap out inputs and append to resets as long as 
        # the difference is greater than a threshold.
        i_swap = 0
        while (i_lowest_scoring_in_use.size > i_swap and
                i_highest_scoring_benched.size > i_swap):
            i_out = self.i_in_use[i_lowest_scoring_in_use[i_swap]]
            i_in = self.i_benched[i_highest_scoring_benched[i_swap]]
            if (candidate_score[i_in] >
                    candidate_score[i_out] + self.score_barrier):
                self.input_mapping[i_in] = self.input_mapping[i_out]
                self.input_mapping[i_out] = -1
                resets.append(self.mapping[i_in])
            else:
                break
            i_swap += 1

        return resets
