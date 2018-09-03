"""
Numba functions that support model.py
"""

from __future__ import print_function
from numba import jit
import numpy as np

import becca.tools as tools


@jit(nopython=True)
def update_sequences(
    feature_activities,
    prefix_activities,
    prefix_occurrences,
    sequence_occurrences,
    sequence_likelihoods,
):
    """
    Update the number of occurrences of each sequence.

    The new sequence activity, n, is
        n = p * f, where
    p is the prefix activities from the previous time step and
    f is the post-feature_activities

    For any given sequence, the probability (sequence_likelihood)
    it will occur on the next time
    step, given the associated goal is selected, is

        f * s / (p + 1)

    where

        f: feature activity
        s: number of sequence occurrences
        p: number of prefix occurrences

    Adding 1 to p prevents badly behaved fractions.
    """
    n_pre_features, n_goals, n_post_features = sequence_occurrences.shape
    # Iterate over post-features
    for j_feature in range(n_post_features):
        if feature_activities[j_feature] > tools.epsilon:
            # Iterate over goals and pre-features
            for i_goal in range(n_goals):
                for i_feature in range(n_pre_features):
                    # These are still the prefix activities from the
                    # previous time step. Together with the current
                    # feature activities, these constitute sequences.
                    if prefix_activities[i_feature, i_goal] > tools.epsilon:
                        sequence_occurrences[
                            i_feature, i_goal, j_feature] += (
                            prefix_activities[i_feature, i_goal] *
                            feature_activities[j_feature])
                        sequence_likelihoods[
                            i_feature, i_goal, j_feature] = (
                            sequence_occurrences[
                                i_feature, i_goal, j_feature] /
                            prefix_occurrences[
                                i_feature, i_goal] + 1)
    return


@jit(nopython=True)
def update_prefixes(
    prefix_decay_rate,
    previous_feature_activities,
    goal_activities,
    prefix_activities,
    prefix_occurrences,
    prefix_uncertainties,
):
    """
    Update the activities and occurrences of the prefixes.

    The new activity of a feature-goal prefix, n,  is
         n = f * g, where
    f is the previous_feature_activities and
    g is the current goal_increase.

    p, the prefix activity, is a decayed version of n.
    """
    n_features, n_goals = prefix_activities.shape
    for i_feature in range(n_features):
        for i_goal in range(n_goals):
            prefix_activities[i_feature, i_goal] *= 1 - prefix_decay_rate

            new_prefix_activity = (previous_feature_activities[i_feature] *
                                   goal_activities[i_goal])
            prefix_activities[i_feature, i_goal] += new_prefix_activity
            prefix_activities[i_feature, i_goal] = min(
                prefix_activities[i_feature, i_goal], 1)

            # Increment the lifetime sum of prefix activity.
            prefix_occurrences[i_feature, i_goal] += (
                prefix_activities[i_feature, i_goal])

            # Adjust uncertainty accordingly.
            prefix_uncertainties[i_feature, i_goal] = 1 / (
                1 + prefix_occurrences[i_feature, i_goal])
    return


@jit(nopython=True)
def update_rewards(
    reward_update_rate,
    reward,
    prefix_credit,
    prefix_rewards,
):
    """
    Assign credit for the current reward to any recently active prefixes.

    Increment the expected reward associated with each prefix.
    The size of the increment is larger when:
        1. the discrepancy between the previously learned and
            observed reward values is larger and
        2. the prefix activity is greater.
    Another way to say this is:
    If either the reward discrepancy is very small
    or the sequence activity is very small, there is no change.
    """
    n_features, n_goals = prefix_rewards.shape
    for i_feature in range(n_features):
        for i_goal in range(n_goals):
            # credit: How much responsibility for this reward is assigned to
            # this prefix?
            credit = prefix_credit[i_feature, i_goal]
            if credit > tools.epsilon:
                # delta: How big is the discrepancy between
                # the observed reward and what has been seen preivously.
                delta = reward - prefix_rewards[i_feature, i_goal]
                if reward > prefix_rewards[i_feature, i_goal]:
                    update_scale = .5
                else:
                    update_scale = 1.
                prefix_rewards[i_feature, i_goal] += (
                    delta * credit * reward_update_rate * update_scale)
    return


@jit(nopython=True)
def update_curiosities(
    curiosity_update_rate,
    prefix_occurrences,
    prefix_curiosities,
    previous_feature_activities,
    feature_activities,
    goal_activities,
    prefix_uncertainties,
):
    """
    Use a collection of factors to increment the curiosity for each prefix.
    """
    n_features, n_goals = prefix_curiosities.shape
    for i_feature in range(n_features):
        for i_goal in range(n_goals):

            # Fulfill curiosity on the previous time step's goals.
            curiosity_fulfillment = (previous_feature_activities[i_feature] *
                                     goal_activities[i_goal])
            prefix_curiosities[i_feature, i_goal] -= curiosity_fulfillment
            prefix_curiosities[i_feature, i_goal] = max(
                prefix_curiosities[i_feature, i_goal], 0)

            # Increment the curiosity based on several multiplicative
            # factors.
            #     curiosity_update_rate : a constant
            #     uncertainty : an estimate of how much is not yet
            #         known about this prefix. It is a function of
            #         the total past occurrences.
            #     feature_activities : The activity of the prefix's feature.
            #         Only increase the curiosity if the feature
            #         corresponding to the prefix is active.
            prefix_curiosities[i_feature, i_goal] += (
                curiosity_update_rate *
                prefix_uncertainties[i_feature, i_goal] *
                feature_activities[i_feature])
    return


@jit(nopython=True)
def predict_features(
    feature_activities,
    sequence_likelihoods,
):
    """
    Make a prediction about which features are going to become active soon,
    conditional on which goals are chosen.

    Parameters
    ----------
    feature_activities: array of floats
    sequence_likelihoods: 3D array of floats
    conditional_predictions: 2D array of floats
        This is updated to represent the new predictions for this time step.
    """
    n_features, n_goals, m_features = sequence_likelihoods.shape
    conditional_predictions = np.zeros((n_goals, m_features))
    for i_feature in range(n_features):
        if feature_activities[i_feature] > tools.epsilon:
            do_nothing_prediction = sequence_likelihoods[i_feature, 1, :]
            # for j_feature in range(2, m_features):
            #     if (do_nothing_prediction[j_feature] >
            #             conditional_predictions[1, j_feature]):
            #         conditional_predictions[1, j_feature] = (
            #             do_nothing_prediction[j_feature])
            for i_goal in range(1, n_goals):
                for j_feature in range(1, m_features):
                    p_sequence = (sequence_likelihoods[
                        i_feature, i_goal, j_feature] -
                        do_nothing_prediction[j_feature])
                    if (p_sequence >
                            conditional_predictions[i_goal, j_feature]):
                        conditional_predictions[i_goal, j_feature] = (
                            p_sequence)
    return conditional_predictions


@jit(nopython=True)
def predict_rewards(
    feature_activities,
    prefix_rewards,
):
    """
    Make a prediction about how much reward will result from each goal.

    For any given prefix, the reward expected to occur on the next time
    step, given the associated goal is selected, is

        f * r

    where

        f: feature activity
        r: prefix rewards

    Parameters
    ----------
    feature_activities: array of floats
    prefix_rewards: 2D array of floats
    conditional_rewards: 2D array of floats
        This is updated to represent the new reward predictions
        for this time step.
    """
    n_features, n_goals = prefix_rewards.shape
    conditional_rewards = np.zeros(n_goals)
    for i_feature in range(2, n_features):
        # Goal[1] is the special "do nothing" goal. It helps to distinguish
        # between reward that is due to a goal and reward that would have
        # been received even if doing nothing.
        do_nothing_reward = prefix_rewards[i_feature, 1]

        # Calculate the expected change in reward for each goal,
        # compared to doing nothing.
        for i_goal in range(1, n_goals):
            expected_reward = (
                feature_activities[i_feature]
                * prefix_rewards[i_feature, i_goal] - do_nothing_reward)
            if expected_reward > conditional_rewards[i_goal]:
                conditional_rewards[i_goal] = expected_reward
    return conditional_rewards


@jit(nopython=True)
def predict_curiosities(
    feature_activities,
    prefix_curiosities,
):
    """
    Make a prediction about how much reward will result from each goal.

    For any given prefix, the reward expected to occur on the next time
    step, given the associated goal is selected, is

        f * r

    where

        f: feature activity
        r: prefix curiosities

    Parameters
    ----------
    feature_activities: array of floats
    prefix_curiosities: 2D array of floats
    conditional_curiosities: 2D array of floats
        This is updated to represent the new reward predictions
        for this time step.
    """
    n_features, n_goals = prefix_curiosities.shape
    conditional_curiosities = np.zeros(n_goals)
    for i_feature in range(n_features):
        # Ignore the first "always on" feature.
        # It doesn't do anything as a goal.
        for i_goal in range(1, n_goals):
            expected_curiosity = (
                feature_activities[i_feature]
                * prefix_curiosities[i_feature, i_goal])
            if expected_curiosity > conditional_curiosities[i_goal]:
                conditional_curiosities[i_goal] = expected_curiosity
    return conditional_curiosities


# This makes use of vectorized numpy calls, so numba is unneccessary here.
# @jit(nopython=True)
def update_fitness(
    feature_fitness,
    prefix_occurrences,
    prefix_rewards,
    prefix_uncertainties,
    sequence_occurrences,
):
    """
    Calculate the fitness of each feature

    Parameters
    ----------
    feature_fitness: array of floats
        The fitness score as of this time step for each of the feature
        inputs to the model. This is modified with calculated values.
    prefix_occurrences: 2D array of floats
    prefix_rewards: 2D array of floats
    prefix_uncertainties: 2D array of floats
    sequence_occurrences: 3D array of floats
    """
    # Calculate the ability of each prefix to predict the features that
    # follow it.
    # Base it on the single most successfully predicted sequence.
    # TODO: debug numba type inference error in this line
    postfeature_prediction_score = (
        np.max(sequence_occurrences, axis=2) /
        (prefix_occurrences + tools.epsilon))
    # Calculate the ability of each prefix to predict reward or punishment.
    reward_prediction_score = np.abs(prefix_rewards)
    prefix_score = postfeature_prediction_score + reward_prediction_score
    # Scale fitness by confidence (1 - uncertainty)
    prefix_fitness = prefix_score * (1 - prefix_uncertainties)
    # Find the maximum fitness for each feature across all prefixes,
    # whether as a prefeature or as a goal.
    prefeature_fitness = np.max(prefix_fitness, axis=1)
    goal_fitness = np.max(prefix_fitness, axis=0)
    feature_fitness = np.maximum(  # noqa: F841
        prefeature_fitness, goal_fitness)

    return


@jit(nopython=True)
def update_reward_credit(
    i_new_goal,
    feature_activities,
    credit_decay_rate,
    prefix_credit,
):
    """
    Update the credit due each prefix for upcoming reward.
    """
    # Age the prefix credit.
    n_features, n_goals = prefix_credit.shape
    for i_feature in range(n_features):
        for i_goal in range(n_goals):
            # Exponential discounting
            prefix_credit[i_feature, i_goal] *= 1 - credit_decay_rate

    # Update the prefix credit.
    if i_new_goal > -1:
        for i_feature in range(n_features):
            # Accumulation strategy:
            # add new credit to existing credit, with a max of 1.
            prefix_credit[i_feature, i_new_goal] += (
                feature_activities[i_feature])
            prefix_credit[i_feature, i_new_goal] = min(
                prefix_credit[i_feature, i_new_goal], 1)
    return
