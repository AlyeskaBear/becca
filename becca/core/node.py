"""
The Node.
"""

import numba as nb
#import numpy as np

@nb.jit(nb.types.Tuple((nb.int32, nb.int32))
        (nb.int32, #node_index, # node parameters
         nb.float64[:], #activity,
         nb.float64[:], #prev_activity,
         nb.float64, #activity_threshold,
         nb.float64, #activity_rate,
         nb.float64[:], #cumulative_activity,
         nb.float64[:], #attempts,
         nb.float64[:], #total_attempts,
         nb.float64[:], #fulfillment,
         nb.float64[:], #unfulfillment,
         nb.float64[:], #choosability,
         nb.float64, #curiosity_rate,
         nb.float64[:], #curiosity,
         #nb.float64, #reward_rate,
         nb.float64[:], #reward,
         nb.float64[:], #value_to_parent,
         #nb.float64, #value_rate,
         nb.int32[:], #element_index,
         nb.int32[:], #sequence_index,
         #nb.float64[:, :], #buds,
         nb.float64, #bud_threshold,
         nb.int32[:], #num_children,
         nb.int32[:, :], #child_indices,
         nb.float64[:], #element_activities, # level parameters
         nb.float64, #parent_activity,
         nb.float64, #prev_parent_activity,
         nb.float64[:], #sequence_activities,
         nb.int32, #num_nodes,
         nb.int32, #num_sequences,
         nb.int32, #max_num_sequences,
         nb.float64[:], #goal_votes,
         nb.float64[:], #element_goals,
         #nb.float64[:], #element_total_weights,
         #nb.float64[:], #element_weighted_values,
         nb.float64[:], #sequence_goals,
         nb.float64, #new_reward,
         nb.float64 #satisfaction
        )
       #)
        ,
        nopython=True)

def step(node_index, # node parameters
         activity,
         prev_activity,
         activity_threshold,
         activity_rate,
         cumulative_activity,
         attempts,
         total_attempts,
         fulfillment,
         unfulfillment,
         choosability,
         curiosity_rate,
         curiosity,
         #reward_rate,
         reward,
         value_to_parent,
         #value_rate,
         element_index,
         sequence_index,
         #buds,
         child_threshold,
         num_children,
         child_indices,
         element_activities, # level parameters
         parent_activity,
         prev_parent_activity,
         sequence_activities,
         num_nodes,
         num_sequences,
         max_num_sequences,
         goal_votes,
         element_goals,
         #element_total_weights,
         #element_weighted_values,
         #min_element_values,
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
    goal_votes :

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
    prev_parent_activity : float
        The activity of the this Node's parent.
    """
    # TODO: update documentation for parameters

    # For the duration of this function call, the processing is focused on
    # node i.
    i = node_index

    # Note what the attempt was from the previous time step.
    # element_goals havent been updated yet, so they still
    # represent what happened on the previous time step.
    # attempts : array of floats
    #     The total number of times the final element of each node 
    #     has been selected as a goal at the same time that the sequence 
    #     corresponding to the node is active.
    attempts[i] = min(prev_parent_activity, element_goals[element_index[i]])

    # Calculate the new node activity.
    # Node activity is either the incoming activity associated with
    # the element index, or the activity from the upstream node,
    # whichever is less.
    #print(i, prev_parent_activity)
    if i == 0:
        activity[i] = 1.
    else:
        activity[i] = min(prev_parent_activity,
                          element_activities[element_index[i]])
    cumulative_activity[i] += activity[i]

    # Fulfill curiosity on the previous time step's attempts.
    # Don't fulfill curiosity on activity. Activity may or may not occur,
    # but attempts the agent can control.
    # curiosity : array of floats
    #     The value associated with selecting this node's element as a goal
    #     when this node's sequence is active.
    curiosity[i] -= attempts[i]
    curiosity[i] = max(0., curiosity[i])

    # Update the curiosity.
    #uncertainty = 1. / (1. + cumulative_activity[i])
    uncertainty = 1. / (1. + fulfillment[i] + unfulfillment[i])
    curiosity[i] += (curiosity_rate * uncertainty *
                     (1. - curiosity[i]) *
                     (1. - satisfaction))

    # fulfillment, unfulfillment : array of floats
    #     All attempts result in either fulfillment or unfulfillment. 
    #     If the node becomes active soon after the attempt, the attempt
    #     is considered fulfilled. The sooner and the larger the activity,
    #     the more complete the fulfillment. All attempts or portions of 
    #     attempts that aren't fulfilled get added to unfulfillment.
    # Fulfill goal attempts when appropriate.
    new_fulfillment = min(activity[i], attempts[i])
    fulfillment[i] += new_fulfillment
    attempts[i] -= new_fulfillment

    # Decay any not-yet fulfilled goals.
    decay = attempts[i] * activity_rate
    unfulfillment[i] += decay
    attempts[i] -= decay

    # choosability : array of floats
    #     The ratio of fulfillment to the cumulative number of past attempts.
    #     Choosabiity is an answer to the question: If I set this node's
    #     element as a goal when its parent node is active, what is the 
    #     likelihood that it will be achieved?
    # Update the node's choosability.
    choosability[i] = fulfillment[i] / (fulfillment[i] + unfulfillment[i])

    # Calculate the combined value of child nodes, 
    # the goal value associated with
    # longer sequences, for which this is a prefix. A child node's
    # value propogates back to this node with a fraction of its value.
    time_penalty = .8
    # choosable_value is the expected value of this node if 
    # it is chosen as a goal when its parent node is active.
    # unchoosable_value is the expected value of this node if
    # it is not chosen as a goal when its parent node is active.
    choosable_children_value = -1.
    unchoosable_children_value = 0.
    #print()
    #print('forward goal for node', i, 'element', element_index[i])
    if num_children[i] > 0:
        for child_index in child_indices[i, :num_children[i]]:
            # choosable_child_value is the expected value of this
            # child, if it this node becomes active, and if the child
            # is then chosen as a goal. 
            choosable_child_value = (value_to_parent[child_index] * 
                                      choosability[child_index])

            # Serendipity is the fraction of the time that the child was
            # stumbled into, unchosen. It is estimated by taking the ratio
            # of the total number of times the child occurred accidentally
            # to the total number of times it could have occurred 
            # accidentally.
            serendipity = ((cumulative_activity[child_index] - 
                            fulfillment[child_index]) /
                           (cumulative_activity[i] - 
                            fulfillment[child_index] - 
                            unfulfillment[child_index]))
            # unchoosable_child_value is the expected value of this child
            # if this node becomes active and the child is *not* chosen
            # as a goal.
            unchoosable_child_value = (value_to_parent[child_index] * 
                                        serendipity)
            # Unchoosable child values are summed. Because they are weighted
            # by serendipity, this is kind of like averaging.
            unchoosable_children_value += unchoosable_child_value
            # Choosable child values are max'ed.
            choosable_children_value = max(choosable_children_value,
                                           choosable_child_value)
            #print('    child element', element_index[child_index],
            #      'tv', value_to_parent[child_index], 
            #      'choo', choosability[child_index],
            #      'choo val', choosable_child_value,
            #      'choo fwd val', choosable_value,
            #      'unchoo val', unchoosable_child_value,
            #      'cum act', cumulative_activity[child_index],
            #      'unchoo fwd val', unchoosable_value)
                  
        children_value = unchoosable_children_value + choosable_children_value
        #print('  unchoo fwd val', unchoosable_value,
        #      'choo fwd val', choosable_value,
        #      'fwd goal', children_value)  
    else:
        children_value = 0.

    # The goal value passed down from the level above.
    if sequence_index[i] != -1:
        top_down_goal = sequence_goals[sequence_index[i]]
    else:
        top_down_goal = 0.

    # total_value is the sum of all sources expected value for this
    # node becoming active: reward, top-down sequence goals and
    # expected value fo children.
    total_value = reward[i] + children_value + top_down_goal
    # value_to_parent : array of floats
    #     The expected value of a node to its parent node.
    #     It is the node's total value  scaled by a time step penalty.
    # The new element value is the maximum of all its constituents.
    value_to_parent[i] = time_penalty * total_value 
    
    # total_goal_value is the strength of this node's vote for its element
    # to be the next goal. This is where curiosity is added in. It doesn't
    # get passed up through parents. It also gets bounded by 0 and 1. 
    total_goal_value = curiosity[i] + choosability[i] * total_value
    total_goal_value = min(max(total_goal_value, 0.), 1.)

    # If this is not the root node, set the relevant sequence activity 
    # for the level and set the relevant element goal.
    if sequence_index[i] != -1:
        sequence_activities[sequence_index[i]] = activity[i]

        # Record weights and totals for calculating the element goal.
        goal_votes[element_index[i]] = max(goal_votes[element_index[i]],
                                           parent_activity * total_goal_value)

        #if parent_activity * total_goal_value > .9:
        #    print()
        #    print('sequence', i, 'element', element_index[i])
        #    print('goal_vote', parent_activity * total_goal_value)
        #    print('total_goal_value', total_goal_value, 
        #          'parent activity', parent_activity)
        #    print('children value', children_value, 'reward', reward[i],
        #          'curiosity', curiosity[i], 'choosability', choosability[i])

    # prev_activity (node activity from the previous time step)
    # is used for the upstream activity instead the node's current
    # activity in order to introduce the one-time-step-per-element
    # temporal structure of the sequence.
    if num_children[i] > 0:
        parent_activity = activity[i]
        prev_parent_activity = prev_activity[i]
        for child_index in child_indices[i, :num_children[i]]:
            num_nodes, num_sequences = (
                step(child_index, # node parameters
                     activity,
                     prev_activity,
                     activity_threshold,
                     activity_rate,
                     cumulative_activity,
                     attempts,
                     total_attempts,
                     fulfillment,
                     unfulfillment,
                     choosability,
                     curiosity_rate,
                     curiosity,
                     #reward_rate,
                     reward,
                     value_to_parent,
                     #value_rate,
                     element_index,
                     sequence_index,
                     #buds,
                     child_threshold,
                     num_children,
                     child_indices,
                     element_activities, # level parameters
                     parent_activity,
                     prev_parent_activity,
                     sequence_activities,
                     num_nodes,
                     num_sequences,
                     max_num_sequences,
                     goal_votes,
                     element_goals,
                     #element_total_weights,
                     #element_weighted_values,
                     #min_element_values,
                     sequence_goals,
                     new_reward,
                     satisfaction))

    # If there is still room for more sequences, grow them.
    else:
    #if num_children[i] == 0:
        if num_sequences < max_num_sequences:
            #print('ca', cumulative_activity[i], 'i', i)
            if cumulative_activity[i] > child_threshold:
                # Create a new set of childes.
                for i_element, _ in enumerate(element_activities):
                    # Don't allow the node to have a child with
                    # the same element index.
                    if i_element != element_index[i]:
                        #print('+++ Creating sequence', num_sequences, 'with', 
                        #      element_index[i])
                        node_index = num_nodes
                        child_indices[i, num_children[i]] = node_index
                        element_index[node_index] = i_element
                        sequence_index[num_nodes] = num_sequences

                        num_children[i] += 1
                        num_nodes += 1
                        num_sequences += 1
                        if num_sequences == max_num_sequences:
                            break

    # Pass the node activity to the next time step.
    prev_activity[i] = activity[i]

    return num_nodes, num_sequences
