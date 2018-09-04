"""
Numba functions that support actor.py
"""
from numba import jit
import numpy as np


@jit(nopython=True)
def calculate_goal_votes(
    conditional_curiosities,
    conditional_predictions,
    conditional_rewards,
    goal_collection,
):
    """
    Assign each goal a value, based on their expected reward, curiosity
    and features.

    Parameters
    ----------
    conditional_predictions: 2D array of floats
    conditional_curiosities,
    conditional_rewards,
    goal_collection: array of floats

    Returns
    -------
    goal_votes: array of floats
        The votes for each feature as the next goal.
    """
    n_goals, n_features = conditional_predictions.shape
    goal_votes = np.zeros(n_goals)

    feature_value = np.zeros(n_goals)
    for i_goal in range(n_goals):
        sum_value = 0
        for i_feature in range(n_features):
            sum_value += (conditional_predictions[i_goal, i_feature] *
                          goal_collection[i_feature])
        feature_value[i_goal] = 1 - 1 / (1 + sum_value)

        goal_votes[i_goal] = (
            conditional_rewards[i_goal] +
            conditional_curiosities[i_goal] +
            feature_value[i_goal])

    # Avoid selecting goals that have been recently selected and are
    # still unfulfilled.
    goal_votes *= 1 - goal_collection
    return goal_votes
