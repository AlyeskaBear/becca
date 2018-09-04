import numpy as np

import becca.actor_numba as nb


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
        n_features : int
        """
        # n_features : int
        #     The maximum number of features.
        self.n_features = n_features + 2

        # goal_collection
        #     The accumulated recent history of goals. These are maintained
        #     so that goals from recent time steps can be translated into
        #     expected value for predicted features.
        self.goal_collection = np.zeros(self.n_features)

        self.goal_decay_rate = .2

    def fulfill(self, feature_activities):
        """
        When a feature is active, goals associated with it are fulfilled.

        Parameters
        ----------
        feature_activities: array of floats
        """
        self.goal_collection *= 1 - feature_activities
        self.goal_collection = np.maximum(0, self.goal_collection)

    def reset(self, resets):
        """
        Reset goals associated with resetted features.

        Parameters
        ----------
        resets: array of ints
            Indices of the goals to reset.
        """
        for i_goal in resets:
            self.goal_collection[i_goal] = 0.

    def choose(
        self,
        conditional_curiosities=None,
        conditional_predictions=None,
        conditional_rewards=None,
    ):
        """
        Using the feature_goal_votes, choose a goal.

        Parameters
        ----------
        conditional_predictions: 2D array of floats
            The expected feature activities, given the selection
            of a goal feature.
        conditional_rewards,
        conditional_curiosities: 1D array of floats
            The expected value or reward and curiosity, given the selection
            of a goal feature.
        """
        self.goal_collection *= 1 - self.goal_decay_rate
        # Choose one goal at each time step, the feature with
        # the largest vote.
        goal_votes = nb.calculate_goal_votes(
            conditional_curiosities,
            conditional_predictions,
            conditional_rewards,
            self.goal_collection,
        )

        # self.previous_feature_goals = self.feature_goal_activities
        goals = np.zeros(self.n_features)
        max_vote = np.max(goal_votes)
        # If there is a tie, randomly select between them.
        i_goal = np.random.choice(np.where(goal_votes == max_vote)[0])
        goals[i_goal] = 1
        self.goal_collection[i_goal] = 1

        return goals, i_goal
