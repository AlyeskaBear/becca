from __future__ import print_function

import numpy as np


class Actor(object):
    """
    Using the predictions from the model, choose next goals.

    Action selection.
    Knowing the
    current active features, goals can be chosen in order to reach
    a desired feature or to maximize reward.
    """
    def __init__(self, n_features, brain):
        """
        Get the Model set up by allocating its variables.

        Parameters
        ----------
        brain : Brain
            The Brain to which this model belongs. Some of the brain's
            parameters are useful in initializing the model.
        n_features : int
            The total number of features allowed in this model.
        """
        # n_features : int
        #     The maximum number of features that the model can expect
        #     to incorporate. Knowing this allows the model to
        #     pre-allocate all the data structures it will need.
        #     Add 2 features/goals that are internal to the model,
        #     An "always on" and a "nothing else is on".
        self.n_features = n_features + 2

        # previous_feature_activities,
        # feature_activities : array of floats
        #     Features are characterized by their
        #     activity, that is, their level of activation at each time step.
        #     Activity can vary between zero and one.
        self.previous_feature_activities = np.zeros(self.n_features)
        self.feature_activities = np.zeros(self.n_features)
        # feature_fitness : array of floats
        #     The predictive fitness of each feature is regularly updated.
        #     This helps determine which features to keep and which to
        #     swap out for new candidates.
        self.feature_fitness = np.zeros(self.n_features)

        # filter: InputFilter
        #     Reduce the possibly large number of inputs to the number
        #     of cables that the Ziptie can handle. Each Ziptie will
        #     have its own InputFilter.
        self.filter = InputFilter(
            n_inputs_final = self.n_inputs,
            verbose=self.verbose,
        )

        # feature_goals,
        # previous_feature_goals,
        # feature_goal_votes : array of floats
        #     Goals can be set for features.
        #     They are temporary incentives, used for planning and
        #     goal selection. These can vary between zero and one.
        #     Votes are used to help choose a new goal each time step.
        self.feature_goal_activities = np.zeros(self.n_features)
        self.previous_feature_goals = np.zeros(self.n_features)
        self.feature_goal_votes = np.zeros(self.n_features)

        # FAIs : array of floats
        #     Feature activity increases.
        #     Of particular interest to us are **increases** in
        #     feature activities. These tend to occur at a
        #     specific point in time, so they are particularly useful
        #     in building meaningful temporal sequences.
        # self.FAIs = np.zeros(self.n_features)

        # prefix_curiosities,
        # prefix_occurrences,
        # prefix_activities,
        # prefix_rewards : 2D array of floats
        # sequence_occurrences : 3D array of floats
        #     The properties associated with each sequence and prefix.
        #     If N is the number of features,
        #     the size of 2D arrays is N**2 and the shape of
        #     3D arrays is N**3. As a heads up, this can eat up
        #     memory as M gets large. They are indexed as follows:
        #         index 0 : feature_1 (past)
        #         index 1 : feature_goal
        #         index 2 : feature_2 (future)
        #     The prefix arrays can be 2D because they lack
        #     information about the resulting feature.
        _2D_size = (self.n_features, self.n_features)
        _3D_size = (self.n_features, self.n_features, self.n_features)
        # Making believe that everything has occurred once in the past
        # makes it easy to believe that it might happen again in the future.
        self.prefix_activities = np.zeros(_2D_size)
        self.prefix_credit = np.zeros(_2D_size)
        self.prefix_occurrences = np.ones(_2D_size)
        self.prefix_curiosities = np.zeros(_2D_size)
        self.prefix_rewards = np.zeros(_2D_size)
        self.prefix_uncertainties = np.zeros(_2D_size)
        self.sequence_occurrences = np.ones(_3D_size)

        # prefix_decay_rate : float
        #     The rate at which prefix activity decays between time steps
        #     for the purpose of calculating reward and finding the outcome.
        self.prefix_decay_rate = .5
        # credit_decay_rate : float
        #     The rate at which the trace, a prefix's credit for the
        #     future reward, decays with each time step.
        self.credit_decay_rate = .35

        # reward_update_rate : float
        #     The rate at which a prefix modifies its reward estimate
        #     based on new observations.
        self.reward_update_rate = 3e-2
        # curiosity_update_rate : float
        #     One of the factors that determines he rate at which
        #     a prefix increases its curiosity.
        self.curiosity_update_rate = 3e-3

    def calculate_fitness():
        """
        Calculate the predictive fitness of all the feature candidates.

        Returns
        -------
        feature_fitness: array of floats
            The fitness of each of the feature candidate inputs to
            the model.
        """

        nb.update_fitness(
            self.feature_fitness,
            self.prefix_occurrences,
            self.prefix_rewards,
            self.prefix_uncertainties,
            self.sequence_occurrences)

        candidate_fitness = self.filter.update_fitness(self.feature_fitness)

        return candidate_fitness

    def update_activities(self, candidate_activities):
        """
        Apply new activities and propogate feature resets, 

        Parameters
        ----------
        candidate_activities: array of floats

        Returns
        -------
        None, but updates class members
        feature_activities: array of floats
        previous_feature_activities: array of floats
        """
        # TODO: incorporate _update_activities() into this
        feature_activities = self.filter.update_activities(
            candidate_activities)

        # Augment the feature_activities with the two internal features,
        # the "always on" (index of 0) and
        # the "null" or "nothing else is on" (index of 1).
        self.previous_feature_activities = self.feature_activities
        self.feature_activities = np.concatenate((
            np.zeros(2), feature_activities))
        self.feature_activities[0] = 1.
        total_activity = np.sum(self.feature_activities[2:])
        inactivity = max(1. - total_activity, 0.)
        self.feature_activities[1] = inactivity
        return

    def update_inputs(self, upstream_resets):
        """
        Add and reset feature inputs as appropriate.

        Parameters
        ----------
        upstream_resets: array of ints
            Indices of the feature candidates to reset.
        """
        resets = self.filter(upstream_resets=upstream_resets)

        # Reset features throughout the model.
        # It's like they never existed.
        for i in resets:
            self.previous_feature_activities[i] = 0.
            self.feature_activities[i] = 0.
            self.feature_fitness[i] = 0.
            self.feature_goal_activities[i] = 0.
            self.previous_feature_goals[i] = 0.
            self.feature_goal_votes[i] = 0.
            # self.FAIs[i] = 0.
            self.prefix_activities[i, :] = 0.
            self.prefix_activities[:, i] = 0.
            self.prefix_credit[i, :] = 0.
            self.prefix_credit[:, i] = 0.
            self.prefix_occurrences[i, :] = 0.
            self.prefix_occurrences[:, i] = 0.
            self.prefix_curiosities[i, :] = 0.
            self.prefix_curiosities[:, i] = 0.
            self.prefix_rewards[i, :] = 0.
            self.prefix_rewards[:, i] = 0.
            self.prefix_uncertainties[i, :] = 0.
            self.prefix_uncertainties[:, i] = 0.
            self.sequence_occurrences[i, :, :] = 0.
            self.sequence_occurrences[:, i, :] = 0.
            self.sequence_occurrences[:, :, i] = 0.

    def step(self, candidate_activities, reward):
        """
        Update the model and choose a new goal.

        Parameters
        ----------
        candidate_activities : array of floats
            The current activity levels of each of the feature candidates.
        reward : float
            The reward reported by the world during the most recent time step.
        """
        self.update_activities(candidate_activities)

        # Update sequences before prefixes.
        nb.update_sequences(
            self.feature_activities,
            self.prefix_activities,
            self.sequence_occurrences)

        nb.update_prefixes(
            self.prefix_decay_rate,
            self.previous_feature_activities,
            self.feature_goal_activities,
            self.prefix_activities,
            self.prefix_occurrences,
            self.prefix_uncertainties)

        nb.update_rewards(
            self.reward_update_rate,
            reward,
            self.prefix_credit,
            self.prefix_rewards)

        nb.update_curiosities(
            self.curiosity_update_rate,
            self.prefix_occurrences,
            self.prefix_curiosities,
            self.previous_feature_activities,
            self.feature_activities,
            self.feature_goal_activities,
            self.prefix_uncertainties)

        self.feature_goal_votes = nb.calculate_goal_votes(
            self.n_features,
            self.prefix_rewards,
            self.prefix_curiosities,
            self.prefix_occurrences,
            self.sequence_occurrences,
            self.feature_activities,
            self.feature_goal_activities)

        # TODO: break this out into a separate object.
        goal_index, max_vote = self._choose_feature_goals()

        nb.update_reward_credit(
            goal_index,
            max_vote,
            self.feature_activities,
            self.credit_decay_rate,
            self.prefix_credit)

        # Trim off the first two elements. The are internal to the model only.
        return self.feature_goal_activities[2:]

    def _choose_feature_goals(self):
        """
        Using the feature_goal_votes, choose a goal.
        """
        # Choose one goal at each time step, the feature with
        # the largest vote.
        self.previous_feature_goals = self.feature_goal_activities
        self.feature_goal_activities = np.zeros(self.n_features)
        max_vote = np.max(self.feature_goal_votes)
        goal_index = 0
        matches = np.where(self.feature_goal_votes == max_vote)[0]
        # If there is a tie, randomly select between them.
        goal_index = matches[np.argmax(
            np.random.random_sample(matches.size))]
        self.feature_goal_activities[goal_index] = 1.

        return goal_index, max_vote


    def visualize(self, brain):
        """
        Make a picture of the model.

        Parameters
        ----------
        brain : Brain
            The brain that this model belongs to.
        """
        viz.visualize(self, brain)
