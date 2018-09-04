import numpy as np

from becca.input_filter import InputFilter
import becca.model_numba as nb


class Model(object):
    """
    Build a predictive model based on sequences of features, goals, reward.

    This version of Becca is model-based, meaning that it builds a
    predictive model of its world in the form of a set of sequences.
    It builds prefixes of the form feature-goal and associates a reward
    and curiosity with each. This is similar to the state-action
    value functions of Q-learning.

    The model also builds feature-goal-feature sequences. These are
    similar to state-action-reward-state-action (SARSA) tuples, as in
    Online Q-Learning using Connectionist Systems" Rummery & Niranjan (1994))
    This formulation allows for prediction, action selection and planning.

    Prediction.
    Knowing the current active features
    and recent goals, both the reward and the resulting features can be
    anticipated.

    Action selection. (performed by the Actor)
    Knowing the
    current active features, goals can be chosen in order to reach
    a desired feature or to maximize reward.

    Planning. (yet to be implemented)
    Feature-goal-feature tuples can
    be chained together to formulate multi-step plans while maximizing
    reward and probability of successfully reaching the goal.
    """
    def __init__(
        self,
        brain=None,
        debug=False,
        n_features=0,
    ):
        """
        Parameters
        ----------
        brain : Brain
            The Brain to which this model belongs. Some of the brain's
            parameters are useful in initializing the model.
        n_features : int
            The total number of features allowed in this model.
        """
        self.debug = debug

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
            n_inputs=self.n_features - 2,
            name='model',
            debug=self.debug,
        )

        # feature_goals,
        # previous_feature_goals,
        # feature_goal_votes : array of floats
        #     Goals can be set for features.
        #     They are temporary incentives, used for planning and
        #     goal selection. These can vary between zero and one.
        #     Votes are used to help choose a new goal each time step.
        self.goal_activities = np.zeros(self.n_features)
        # self.previous_feature_goals = np.zeros(self.n_features)
        # self.feature_goal_votes = np.zeros(self.n_features)

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
        #         index 0 : feature_1 (past or pre-feature)
        #         index 1 : feature_goal
        #         index 2 : feature_2 (future or post-feature)
        #     The prefix arrays can be 2D because they lack
        #     information about the resulting feature.
        _2D_size = (self.n_features, self.n_features)
        _3D_size = (self.n_features, self.n_features, self.n_features)
        # Making believe that everything has occurred once in the past
        # makes it easy to believe that it might happen again in the future.
        self.conditional_rewards = np.zeros(self.n_features)
        self.conditional_curiosities = np.zeros(self.n_features)
        self.conditional_predictions = np.zeros(_2D_size)
        self.prefix_activities = np.zeros(_2D_size)
        self.prefix_credit = np.zeros(_2D_size)
        self.prefix_occurrences = np.ones(_2D_size)
        self.prefix_curiosities = np.zeros(_2D_size)
        self.prefix_rewards = np.zeros(_2D_size)
        self.prefix_uncertainties = np.zeros(_2D_size)
        self.sequence_occurrences = np.zeros(_3D_size)
        self.sequence_likelihoods = np.zeros(_3D_size)

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
        # Update feature_activities and previous_feature_activities
        self.update_activities(candidate_activities)

        # Update sequences before prefixes.
        nb.update_sequences(
            self.feature_activities,
            self.prefix_activities,
            self.prefix_occurrences,
            self.sequence_occurrences,
            self.sequence_likelihoods,
        )

        nb.update_prefixes(
            self.prefix_decay_rate,
            self.previous_feature_activities,
            self.goal_activities,
            self.prefix_activities,
            self.prefix_occurrences,
            self.prefix_uncertainties,
        )

        nb.update_rewards(
            self.reward_update_rate,
            reward,
            self.prefix_credit,
            self.prefix_rewards,
        )

        nb.update_curiosities(
            self.curiosity_update_rate,
            self.prefix_occurrences,
            self.prefix_curiosities,
            self.previous_feature_activities,
            self.feature_activities,
            self.goal_activities,
            self.prefix_uncertainties,
        )

        self.conditional_predictions = nb.predict_features(
            self.feature_activities,
            self.sequence_likelihoods,
        )

        self.conditional_rewards = nb.predict_rewards(
            self.feature_activities,
            self.prefix_rewards,
        )

        self.conditional_curiosities = nb.predict_curiosities(
            self.feature_activities,
            self.prefix_curiosities,
        )

        return (
            self.conditional_predictions,
            self.conditional_rewards,
            self.conditional_curiosities)

    def update_activities(self, candidate_activities):
        """
        Apply new activities,

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

    def calculate_fitness(self):
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

        candidate_fitness = self.filter.update_fitness(
            self.feature_fitness[2:])

        return candidate_fitness

    def update_inputs(self, upstream_resets):
        """
        Add and reset feature inputs as appropriate.

        Parameters
        ----------
        upstream_resets: array of ints
            Indices of the feature candidates to reset.

        Returns
        -------
        resets: array of ints
            Indices of the features that were reset.
        """
        resets = self.filter.update_inputs(upstream_resets=upstream_resets)
        # Account for the model's 2 internal features.
        model_resets = [i_reset + 2 for i_reset in resets]
        # Reset features throughout the model.
        # It's like they never existed.
        for i in model_resets:
            self.previous_feature_activities[i] = 0.
            self.feature_activities[i] = 0.
            self.feature_fitness[i] = 0.
            self.goal_activities[i] = 0.
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
            self.sequence_likelihoods[i, :, :] = 0.
            self.sequence_likelihoods[:, i, :] = 0.
            self.sequence_likelihoods[:, :, i] = 0.

        return resets

    def update_goals(self, goals, i_new_goal):
        """
        Given a set of goals, record and execute them.

        Parameters
        ----------
        goals: array of floats
            The current set of goal activities.
        i_new_goal: int
            The index of the most recent feature goal.

        Returns
        -------
        feature_pool_goals: array of floats
        """
        self.goal_activities = goals
        nb.update_reward_credit(
            i_new_goal,
            self.feature_activities,
            self.credit_decay_rate,
            self.prefix_credit)

        # Trim off the first two elements.
        # The are internal to the model only.
        feature_pool_goals = self.filter.project_activities(
            self.goal_activities[2:])
        return feature_pool_goals
