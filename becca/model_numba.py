"""
Numba functions that support model.py
"""
from numba import jit
import numpy as np


@jit(nopython=True)
def update_sequences(
    live_features,
    new_FAIs,
    prefix_activities,
    sequence_occurrences):
    """
    Update the number of occurrences of each sequence.

    The new sequence activity, n, is
        n = p * f, where
    p is the prefix activities from the previous time step and
    f is the new_FAIs
    """
    small = .1
    for j_feature in live_features:
        if (new_FAIs[j_feature] < small):
            continue
        for i_goal in live_features:
            for i_feature in live_features:
                sequence_occurrences[i_feature][i_goal][j_feature] += (
                    prefix_activities[i_feature][i_goal] *
                    new_FAIs[j_feature])
    return

@jit(nopython=True)
def update_prefixes(
    live_features,
    prefix_decay_rate,
    previous_FAIs,
    FGIs,
    prefix_activities,
    prefix_activities_base,
    prefix_activities_age,
    prefix_occurrences):
    """
    Update the activities and occurrences of the prefixes.

    The new activity of a feature-goal prefix, n,  is
         n = f * g, where
    f is the previous_FAI and
    g is the current goal_increase.
    p, the prefix activity, is a decayed version of n.
    """
    for i_feature in live_features:
        for i_goal in live_features:
            new_prefix_activity = (
                previous_FAIs[i_feature] *
                FGIs[i_goal])

            prefix_activities_age[i_feature][i_goal] += 1.
            if new_prefix_activity > prefix_activities[i_feature][i_goal]:
                prefix_activities_base[i_feature][i_goal] = new_prefix_activity
                prefix_activities_age[i_feature][i_goal] = 0.

            # Decay activities hyperbolically.
            prefix_activities[i_feature][i_goal] = (
                prefix_activities_base[i_feature][i_goal] *(
                (1. / (1. + 2. * prefix_activities_age[i_feature][i_goal]))))

            # Use a leaky integration filter to give the prefix
            # activity a bit of a lifetime. This gives it a reward
            # credit and lets it learn time-delayed terminal features.
            # TODO: Add hyperbolic discounting
            #prefix_activities[i_feature][i_goal] = np.maximum(
            #    new_prefix_activity,
            #    prefix_activities[i_feature][i_goal] *
            #    (1. - prefix_decay_rate))

            prefix_occurrences[i_feature][i_goal] += (
                prefix_activities[i_feature][i_goal])
    return


@jit(nopython=True)
def update_rewards(
    live_features,
    reward_update_rate,
    reward,
    prefix_activities,
    prefix_occurrences,
    prefix_credit,
    prefix_rewards):
    """
    Assign credit for the current reward to any recently active prefixes.
    """
    for i_feature in live_features:
        for i_goal in live_features:
            # Adjust the reward update rate so that the node adjusts
            # very quickly to the first few exposures and then gradually
            # more slowly to subsequent ones.
            # This tends to decrease training
            # time. It's an agressive learning strategy, and so is
            # prone to initial error, but it allows
            # for later experiences to refine the reward estimate and
            # eventually correct that error out.
            mod_reward_rate = max(
                reward_update_rate,
                1. / (1. + prefix_occurrences[i_feature][i_goal]))
            #mod_reward_rate = reward_update_rate

            # Increment the expected reward associated with each prefix.
            # The size of the increment is larger when:
            #     1. the discrepancy between the previously learned and
            #         observed reward values is larger and
            #     2. the prefix activity is greater.
            # Another way to say this is:
            # If either the reward discrepancy is very small
            # or the sequence activity is very small, there is no change.
            prefix_rewards[i_feature][i_goal] += (
                (reward - prefix_rewards[i_feature][i_goal]) *
                prefix_credit[i_feature][i_goal] *
                #prefix_activities[i_feature][i_goal] *
                mod_reward_rate)
    return


@jit(nopython=True)
def update_curiosities(
    satisfaction,
    live_features,
    curiosity_update_rate,
    prefix_occurrences,
    prefix_curiosities,
    FAIs,
    previous_FAIs,
    FGIs,
    feature_goal_activities):
    """
    Use a collection of factors to increment the curiosity for each prefix.
    """
    for i_feature in live_features:
        for i_goal in live_features:

            # Fulfill curiosity on the previous time step's goals.
            curiosity_fulfillment = previous_FAIs[i_feature] * FGIs[i_goal] 
            #prefix_curiosities[i_feature][i_goal] -= (
            #    feature_goal_activities[i_goal])
            prefix_curiosities[i_feature][i_goal] -= curiosity_fulfillment
            prefix_curiosities[i_feature][i_goal] = max(
                prefix_curiosities[i_feature][i_goal], 0.)

            # Increment the curiosity based on several multiplicative
            # factors.
            #     curiosity_update_rate : a constant
            #     uncertainty : an estimate of how much is not yet
            #         known about this prefix. It is a function of
            #         the total past occurrences.
            #     FAIs : The activity of the prefix.
            #         Only increase the curiosity if the feature
            #         corresponding to the prefix is active.
            #     1 - curiosity : This is a squashing factor that
            #         ensures that curiosity will asymptotically approach 1.
            #     1 - satisfaction : This is a scaling factor to account for
            #         contentment in the agent. If the agent is consistently
            #         getting high rewards, there is little need
            #         to be curious.
            uncertainty = 1. / (1. + prefix_occurrences[i_feature][i_goal])
            prefix_curiosities[i_feature][i_goal] += (
                curiosity_update_rate *
                uncertainty *
                FAIs[i_feature] *
                (1. - prefix_curiosities[i_feature][i_goal]) *
                (1. - satisfaction))
            ''' 
            print(' ')
            print('i_feature', i_feature)
            print('i_goal', i_goal)
            print('previous_FAIs[i_feature]', previous_FAIs[i_feature])
            print('FGIs[i_goal]', FGIs[i_goal])
            print('curiosity_fulfillment', curiosity_fulfillment)
            print('prefix_curiosities[i_feature][i_goal]',
                  prefix_curiosities[i_feature][i_goal])
            print('uncertainty', uncertainty)
            print('prefix_occurrences[i_feature][i_goal]',
                  prefix_occurrences[i_feature][i_goal])
            print('curiosity_update_rate', curiosity_update_rate)
            print('FAIs[i_feature]', FAIs[i_feature])
            print('satisfaction', satisfaction)
            '''
    return


@jit(nopython=True)
def calculate_goal_votes(
    num_features,
    live_features,
    time_since_goal,
    jumpiness,
    prefix_goal_votes,
    prefix_credit,
    prefix_rewards,
    prefix_curiosities,
    prefix_occurrences,
    sequence_occurrences,
    FAIs,
    feature_goal_activities):
    """
    Let each prefix cast a vote for its goal, based on its value.
    """
    # For each prefix and its corresponding sequences calculate
    # the expected value, v, of the goal.
    #
    #     v = a * (r + c + s), where
    # a is the activity of the prefix's feature,
    # r is the prefix's reward,
    # c is the prefix's curiosity, and
    # s is the overall expected value of the prefix's sequences.
    #
    #     s = sum((o / p) * (g + t)) / sum(o / p), where
    # o is the number of occurrences of the sequence,
    # p is the number of occurrences of the prefix,
    # g is the goal value of the sequence's terminal feature, and
    # t is the top-down plan value of the sequence's terminal feature.
    #
    # For each goal, track the largest value that is calculated and
    # treat it as a vote for that goal.
    small = .1
    feature_goal_votes = np.zeros(num_features)
    for i_feature in live_features:
        if (FAIs[i_feature] < small):
            continue
        for i_goal in live_features:

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
            goal_vote = FAIs[i_feature] * (
                prefix_rewards[i_feature][i_goal] +
                prefix_curiosities[i_feature][i_goal] +
                time_since_goal * jumpiness)# +
                # TODO reinstate
                #sequence_value)

            prefix_goal_votes[i_feature][i_goal] = goal_vote

            # Compile the maximum goal votes for action selection.
            feature_goal_votes[i_goal] = max(
                feature_goal_votes[i_goal], goal_vote)
    return feature_goal_votes


@jit(nopython=True)
def update_reward_credit(
    live_features,
    i_new_goal,
    max_vote,
    prefix_goal_votes,
    prefix_credit,
    prefix_credit_base,
    prefix_credit_age,
    ):
    """
    Update the credit due each prefix for upcoming reward.
    """
    # Age the prefix credit.
    for i_feature in live_features:
        for i_goal in live_features:
            prefix_credit_age[i_feature][i_goal] += 1.
            prefix_credit[i_feature][i_goal] = (
                prefix_credit_base[i_feature][i_goal] * (1. / (
                1. + 4. * prefix_credit_age[i_feature][i_goal])))

    # Update the prefix credit.
    if i_new_goal > -1:
        for i_feature in live_features:
            # Measure credit as a fraction of the largest vote for the
            # current goal.
            new_credit_base = (prefix_goal_votes[i_feature][i_new_goal] /
                               max_vote)
            new_credit_base = max(new_credit_base, 0.)
            # If that credit is larger than the current credit, replace it.
            if new_credit_base > prefix_credit[i_feature][i_new_goal]:
                prefix_credit[i_feature][i_new_goal] = new_credit_base
                prefix_credit_base[i_feature][i_new_goal] = new_credit_base
                prefix_credit_age[i_feature][i_new_goal] = 0.
    return
