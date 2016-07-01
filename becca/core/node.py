"""
The Node.
"""

import numba as nb

@nb.jit(nb.types.Tuple((nb.int32, nb.int32))
        (nb.int32, #node_index, # node parameters
         nb.float64[:], #activity,
         nb.float64[:], #prev_activity,
         nb.float64, #activity_threshold,
         nb.float64, #activity_rate,
         nb.float64[:], #cumulative_activity,
         nb.float64, #curiosity_rate,
         nb.float64[:], #curiosity,
         nb.float64, #reward_rate,
         nb.float64[:], #reward,
         nb.float64[:], #total_value,
         nb.float64, #value_rate,
         nb.int32[:], #element_index,
         nb.int32[:], #sequence_index,
         nb.float64[:, :], #buds,
         nb.float64, #bud_threshold,
         nb.int32[:], #num_branches,
         nb.int32[:, :], #branch_indices,
         nb.float64[:], #element_activities, # level parameters
         nb.float64, #upstream_activity,
         nb.float64[:], #sequence_activities,
         nb.int32, #num_nodes,
         nb.int32, #num_sequences,
         nb.int32, #max_num_sequences,
         nb.float64[:], #element_goals,
         nb.float64[:], #sequence_goals,
         nb.float64, #new_reward,
         nb.float64 #satisfaction
        ), 
        nopython=True)
def step(node_index, # node parameters
         activity,
         prev_activity,
         activity_threshold,
         activity_rate,
         cumulative_activity,
         curiosity_rate,
         curiosity,
         reward_rate,
         reward,
         total_value,
         value_rate,
         element_index,
         sequence_index,
         buds,
         bud_threshold,
         num_branches,
         branch_indices,
         element_activities, # level parameters
         upstream_activity,
         sequence_activities,
         num_nodes,
         num_sequences,
         max_num_sequences,
         element_goals,
         sequence_goals,
         new_reward,
         satisfaction
        ):
    """
    Update the Node with the most recent element activities.

    Parameters
    ----------
    element_activities : array of floats
        The most recent of observed element activities.
    element_goals : array of floats
        The goals (internal reward due to planning and incentives)
        associated with each element.
    new_reward : float
        The current value of the reward.
    satisfaction : float
        The state of satisfaction of the brain. A filtered version
        of reward.
    sequence_activities : array of floats
        The set of activities associated with each of the sequences (Nodes)
        in the Level.
    sequence_goals : array of floats
        The goals (internal reward due to planning and incentives)
        associated with each sequence (Node).
    upstream_activity : float
        The activity of the this Node's parent.
    """
    # TODO: update documentation for parameters

    # For the duration of this function call, the processing is focused on
    # node i.
    i = node_index

    # Calculate the new node activity.
    # Node activity is either the incoming activity associated with
    # the element index, or the activity from the upstream node,
    # whichever is less.
    activity[i] = min(upstream_activity, element_activities[element_index[i]])

    # If the decayed activity from the previous time step was greater,
    # revert to that.
    activity[i] = max(prev_activity[i] * (1. - activity_rate), activity[i])
    cumulative_activity[i] += activity[i]

    # Set the relevant sequence activity for the Level.
    if sequence_index[i] != -1:
        sequence_activities[sequence_index[i]] = activity[i]

    # Update the reward estimate.
    #
    # Increment the expected reward value associated with each sequence.
    # The size of the increment is larger when:
    #     1. the discrepancy between the previously learned and
    #         observed reward values is larger and
    #     2. the sequence activity is greater.
    # Another way to say this is:
    # If either the reward discrepancy is very small
    # or the sequence activity is very small, there is no change.
    reward[i] += (new_reward - reward[i]) * activity[i] * reward_rate

    # Fulfill curiosity.
    curiosity[i] *= 1. - activity[i]
    # Update the curiosity.
    uncertainty = 1. / (1. + cumulative_activity[i])
    curiosity[i] += (curiosity_rate * uncertainty *
                     (1. - curiosity[i]) *
                     (1. - satisfaction))

    # Calculate the forward goal, the goal value associated with
    # longer sequences, for which this is a prefix. A child node's
    # value propogates back to this node with a fraction of its
    # value, given by:
    #
    #       fraction = (child node's cumulative activity + bud threshold)
    #                   ________________________________________________
    #                         this node's cumulative activity
    #
    # bud_threshold is added to the numerator to account for the fact
    # that parent nodes get that much of a head start on their activity count.
    # The forward goal is the maximum propogated goal across
    # all child nodes.
    forward_goal = -1
    for branch_index in branch_indices[i, :num_branches[i]]:
        node_value = total_value[branch_index] * (
            (cumulative_activity[branch_index] + bud_threshold) /
            cumulative_activity[i])
        if node_value > forward_goal:
            forward_goal = node_value

    # The goal value passed down from the level above.
    if sequence_index[i] != -1:
        top_down_goal = sequence_goals[sequence_index[i]]
    else:
        top_down_goal = 0.
    # The new element value is the maximum of all its constituents.
    total_value_ceil = max(max(forward_goal, top_down_goal),
                           max(curiosity[i], reward[i]))
    new_total_value = total_value_ceil * (1. - activity[i])
    total_value[i] = max(new_total_value, total_value[i] * (1. - value_rate))
    # Pass the new element goal down.
    element_goals[element_index] = max(element_goals[element_index[i]],
                                       total_value[i])

    # prev_activity (node activity from the previous time step)
    # is used for the upstream activity instead the node's current
    # activity in order to introduce the one-time-step-per-element
    # temporal structure of the sequence.
    upstream_activity = prev_activity[i]
    for node_index in branch_indices[i, :num_branches[i]]:
        num_nodes, num_sequences = (
            step(node_index, # node parameters
                 activity,
                 prev_activity,
                 activity_threshold,
                 activity_rate,
                 cumulative_activity,
                 curiosity_rate,
                 curiosity,
                 reward_rate,
                 reward,
                 total_value,
                 value_rate,
                 element_index,
                 sequence_index,
                 buds,
                 bud_threshold,
                 num_branches,
                 branch_indices,
                 element_activities, # level parameters
                 upstream_activity,
                 sequence_activities,
                 num_nodes,
                 num_sequences,
                 max_num_sequences,
                 element_goals,
                 sequence_goals,
                 new_reward,
                 satisfaction))

    # If there is still room for more sequences, grow them.
    if num_sequences < max_num_sequences:
        # Update each of the child buds.
        #print(element_activities.shape, self.buds.shape)
        # Again, use prev_activity to get one-time-step-per-element.
        buds[i, :] += prev_activity[i] * element_activities
        # Don't allow the node to have a child with the same element index,
        buds[i, element_index[i]] = 0.
        # or with nodes that they already have.
        for branch_index in branch_indices[i, :num_branches[i]]:
            buds[i, element_index[branch_index]] = 0.

        # Create a new branch when appropriate.
        for bud_index, bud_value in enumerate(buds[i, :]):
            if bud_value > bud_threshold:
                # Populate values for the new branch.
                #print('nb',  num_branches[i],
                #      'i', i,
                #      'nn', num_nodes,
                #      'bi', bud_index)
                branch_indices[i, num_branches[i]] = num_nodes
                element_index[num_nodes] = bud_index
                sequence_index[num_nodes] = num_sequences
                reward[num_nodes] = reward[i]

                buds[i, bud_index] = 0.
                num_branches[i] += 1
                num_sequences += 1
                num_nodes += 1

    # Pass the node activity to the next time step.
    prev_activity[i] = activity[i]

    return num_nodes, num_sequences
