"""
Numba functions that support model.py
"""

from __future__ import print_function
from numba import jit
import numpy as np


@jit(nopython=True)
def calculate_goal_votes(
    num_features,
    live_features,
    prefix_rewards,
    prefix_curiosities,
    prefix_occurrences,
    sequence_occurrences,
    feature_activities,
    feature_goal_activities,
):
    """
    Let each prefix cast a vote for its goal, based on its value.

    For each prefix and its corresponding sequences calculate
    the expected value, v, of the goal.

            v = a * (r + c + s), where
        a is the activity of the prefix's feature,
        r is the prefix's reward,
        c is the prefix's curiosity, and
        s is the overall expected value of the prefix's sequences.

            s = sum((o / p) * (g + t)) / sum(o / p), where
        o is the number of occurrences of the sequence,
        p is the number of occurrences of the prefix,
        g is the goal value of the sequence's terminal feature, and
        t is the top-down plan value of the sequence's terminal feature.

    For each goal, track the largest value that is calculated and
    treat it as a vote for that goal.
    """
    small = .1
    feature_goal_votes = np.zeros(num_features)
    for i_feature in live_features:
        for i_goal in live_features:
            if feature_activities[i_feature] < small:
                goal_vote = -2.

            else:
                ## Hold out this code for now. As far as I know, it works.
                ## I'll be able to tell with more certainty after additional
                ## testing.
                # Add up the value of sequences.
                weighted_values = 1.
                total_weights = 1.
                for j_feature in live_features:
                    weight = (
                        sequence_occurrences[i_feature][i_goal][j_feature]*
                        prefix_occurrences[i_feature][i_goal])
                    weighted_values += (
                        weight * feature_goal_activities[j_feature])
                    total_weights += weight
                sequence_value = weighted_values / total_weights
                # Add up the other value components.
                goal_vote = feature_activities[i_feature] * (
                    prefix_rewards[i_feature][i_goal] +
                    prefix_curiosities[i_feature][i_goal])# +
                    #sequence_value)

            # Compile the maximum goal votes for action selection.
            if goal_vote > feature_goal_votes[i_goal]:
                feature_goal_votes[i_goal] = goal_vote

    return feature_goal_votes
