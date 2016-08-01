"""
The Node.

This module contains a couple of functions that do most of the heavy lifting. 
They use numba to compile int C, so they run *really* fast. Even faster
than vectorized numpy.

step() : This one is repsponsible for the one-time step advancement
    of a node. It is the heart of BECCA. It recursively calls all the 
    other nodes within a level too. Because recurive calls within numba
    is a relatively new feature, this demands numba 27 or later. 
    It is also the reason that the function signature is explicitly 
    typed, instead of just using an @autojit decorator.

update_rewards() : This function updates the reward associated with
    each of the nodes. It manages the credit assignment and reward updates
    across all nodes--some of the most critical aspects of BECCA's 
    reinforcement learning algorithm.
"""

import numba as nb
import numpy as np

@nb.jit(nb.types.Tuple((nb.int32, nb.int32))
        (nb.int32, #node_index
         nb.float64[:], #activity,
         nb.float64[:], #prev_activities,
         nb.float64, #activity_rate,
         nb.float64[:], #cumulative_activities,
         nb.float64[:], #attempts,
         nb.float64[:], #fulfillment,
         nb.float64[:], #unfulfillment,
         nb.float64, #curiosity_rate,
         nb.float64[:], #curiosity,
         nb.float64[:], #reward,
         nb.float64[:], #value_to_parent,
         nb.int32[:], #element_index,
         nb.int32[:], #sequence_index,
         nb.int32[:], #sequence_length,
         nb.float64, #bud_threshold,
         nb.int32[:], #num_children,
         nb.int32[:, :], #child_indices,
         nb.int32[:], #parent_index,
         nb.float64[:], #element_activities
         nb.int32, #num_active_elements,
         nb.float64, #parent_activity,
         nb.float64, #prev_parent_activity,
         nb.float64[:], #sequence_activities,
         nb.int32, #num_nodes,
         nb.int32, #num_sequences,
         nb.int32, #max_num_sequences,
         nb.int32, #max_num_nodes,
         nb.float64[:], #goal_votes,
         nb.int32[:], #responsible_nodes,
         nb.float64[:], #element_goals,
         nb.float64[:], #sequence_goals,
         nb.float64, #new_reward,
         nb.float64 #satisfaction
        ), nopython=True)

def step(node_index, # node parameters
         activity,
         prev_activities,
         activity_rate,
         cumulative_activities,
         attempts,
         fulfillment,
         unfulfillment,
         curiosity_rate,
         curiosity,
         reward,
         value_to_parent,
         element_index,
         sequence_index,
         sequence_length,
         sequence_threshold,
         num_children,
         child_indices,
         parent_index,
         element_activities, # level parameters
         num_active_elements,
         parent_activity,
         prev_parent_activity,
         sequence_activities,
         num_nodes,
         num_sequences,
         max_num_sequences,
         max_num_nodes,
         goal_votes,
         responsible_nodes,
         element_goals,
         sequence_goals,
         new_reward,
         satisfaction
        ):
    """
    Update the Node with the most recent element activities.

    The step() function carries out BECCA's feature creation,
    model learning and action selection. These are all incremental
    methods, meaning that they take the new inputs at each time step
    and update all the internal parameters and estimates incrementally.

    I apologize for the ridiculous number of variables passed into this
    function. When I figure out a more elegant way to do this, I'll 
    implement it. Most of the input arguments are defined in level.py.
    The few that aren't are defined here.

    Parameters
    ----------------
    new_reward : float
        The current value of the reward.
    satisfaction : float
        The state of satisfaction of the brain. A filtered version
        of reward.
    parent_activity, prev_parent_activity : float
        The activity of the this node's parent from the current and previous
        time step..
    """
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
    if i == 0:
        activity[i] = 1.
    else:
        activity[i] = min(prev_parent_activity,
                          element_activities[element_index[i]])
    cumulative_activities[i] += activity[i]

    # Fulfill curiosity on the previous time step's attempts.
    # Don't fulfill curiosity on activity. Activity may or may not occur,
    # but attempts the agent can control.
    # curiosity : array of floats
    #     The value associated with selecting this node's element as a goal
    #     when this node's sequence is active.
    curiosity[i] -= attempts[i]
    curiosity[i] = max(0., curiosity[i])

    # Update the curiosity.
    #uncertainty = 1. / (1. + cumulative_activities[i])
    uncertainty = 1. / (1. + fulfillment[i] + unfulfillment[i])
    curiosity[i] += (curiosity_rate * 
                     uncertainty * 
                     parent_activity *
                     (1. - curiosity[i]) *
                     (1. - satisfaction))

    # Fulfill goal attempts when appropriate.
    new_fulfillment = min(activity[i], attempts[i])
    fulfillment[i] += new_fulfillment
    attempts[i] -= new_fulfillment

    # Decay any not-yet fulfilled goals.
    decay = attempts[i] * activity_rate
    unfulfillment[i] += decay
    attempts[i] -= decay

    # The goal value passed down from the level above.
    if sequence_index[i] != -1:
        top_down_goal = sequence_goals[sequence_index[i]]
    else:
        top_down_goal = 0.

    # total_value is the sum of all sources expected value for this
    # node becoming active: reward, top-down sequence goals and
    # expected value fo children.
    #total_value = reward[i] + children_value + top_down_goal
    #total_value = reward[i] + top_down_goal
    # value_to_parent : array of floats
    #     The expected value of a node to its parent node.
    #     It is the node's total value  scaled by a time step penalty.
    # The new element value is the maximum of all its constituents.
    #value_to_parent[i] = time_penalty * total_value

    # total_goal_value is the strength of this node's vote for its element
    # to be the next goal. This is where curiosity is added in. It doesn't
    # get passed up through parents. It also gets bounded by 0 and 1.
    #total_goal_value = curiosity[i] + choosability[i] * total_value
    #total_goal_value = curiosity[i] + reward[i]

    # If this is not the root node, set the relevant sequence activity
    # for the level and set the relevant element goal.
    if sequence_index[i] != -1:
        sequence_activities[sequence_index[i]] = activity[i]

    if sequence_length[i] > 1:
        # Record weights and totals for calculating the element goal.
        element_goal_vote = parent_activity * (curiosity[i] + 
                                               reward[i] +
                                               top_down_goal)
        if element_goal_vote > goal_votes[element_index[i]]:
            goal_votes[element_index[i]] = element_goal_vote
            responsible_nodes[element_index[i]] = i

    # prev_activities (node activity from the previous time step)
    # is used for the upstream activity instead the node's current
    # activity in order to introduce the one-time-step-per-element
    # temporal structure of the sequence.
    if num_children[i] > 0:
        parent_activity = activity[i]
        prev_parent_activity = prev_activities[i]
        for child_index in child_indices[i, :num_children[i]]:
            if element_index[child_index] < num_active_elements:
                num_nodes, num_sequences = (
                    step(child_index, # node parameters
                         activity,
                         prev_activities,
                         activity_rate,
                         cumulative_activities,
                         attempts,
                         fulfillment,
                         unfulfillment,
                         curiosity_rate,
                         curiosity,
                         reward,
                         value_to_parent,
                         element_index,
                         sequence_index,
                         sequence_length,
                         sequence_threshold,
                         num_children,
                         child_indices,
                         parent_index,
                         element_activities, # level parameters
                         num_active_elements,
                         parent_activity,
                         prev_parent_activity,
                         sequence_activities,
                         num_nodes,
                         num_sequences,
                         max_num_sequences,
                         max_num_nodes,
                         goal_votes,
                         responsible_nodes,
                         element_goals,
                         sequence_goals,
                         new_reward,
                         satisfaction))

    # If there is still room for more sequences, grow them.
    else: #if num_children[i] == 0:
        #if num_sequences < max_num_sequences and num_nodes < max_num_nodes:
        if num_sequences < max_num_sequences:
            mod_threshold = sequence_threshold * (
                1. + num_sequences / max_num_sequences)
            if cumulative_activities[i] > mod_threshold:
                # Assign this node its own output sequence, now that it
                # has been observed enough times.
                sequence_index[i] = num_sequences
                num_sequences += 1
                # Create a new set of children.
                #for i_element, _ in enumerate(element_activities):
                #    # Don't allow the node to have a child with
                #    # the same element index.
                #    if i_element != element_index[i]:
                #        #print('+++ Creating sequence', num_sequences, 'with',
                #        #      element_index[i])
                #        node_index = num_nodes
                #        sequence_length[node_index] = sequence_length[i] + 1
                #        child_indices[i, num_children[i]] = node_index
                #        element_index[node_index] = i_element
                #        parent_index[node_index] = i
                #        num_children[i] += 1
                #        num_nodes += 1
                #        if num_nodes == max_num_nodes:
                #            break

    # Pass the node activity to the next time step.
    prev_activities[i] = activity[i]

    return num_nodes, num_sequences


@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:, :], nb.int32))
        (nb.float64[:], #node_reward
         nb.float64, #reward
         nb.float64, #reward_rate
         nb.int32, #reward_trace_length
         nb.float64[:], #decay
         nb.float64[:, :], #trace_history
         nb.int32, #trace_history_length
         nb.int32, #trace_index
         nb.float64[:], #cumulative_activities
         nb.float64[:], #node_activities
         nb.int32[:], #element_index
         nb.int32[:], #parent_index
         nb.int32, #goal_index
         nb.int32, #num_nodes
        ), nopython=True)

def update_rewards(node_reward,
                   reward,
                   reward_rate,
                   trace_length,
                   trace_decay,
                   trace_history,
                   trace_history_length,
                   trace_index,
                   cumulative_activities,
                   node_activities,
                   element_index,
                   parent_index,
                   goal_index,
                   num_nodes
                  ):
    """
    Update node reward estimates.
    """
    t_history = np.zeros(trace_length, np.int32)
    for t in range(trace_length):
        # Cycle through the node activity history, starting with the
        # most recent time step and working backward.
        t_past = trace_index - t
        if t_past < 0:
            t_past += trace_history_length
        t_history[t] = t_past
    
    for i in range(num_nodes):
        # Adjust the reward update rate so that the node adjusts
        # very quickly to the first few exposures and then gradually
        # more slowly to subsequent ones. This tends to decrease training
        # time. It's an agressive learning strategy, and so is
        # prone to initial error, but it allows
        # for later experiences to refine the reward estimate and
        # eventually correct that error out.
        mod_reward_rate = max(reward_rate, 
                              1. / (1. + cumulative_activities[i]))
        for t in range(trace_length):
            # Increment the expected reward value associated with each sequence.
            # The size of the increment is larger when:
            #     1. the discrepancy between the previously learned and
            #         observed reward values is larger and
            #     2. the sequence activity is greater.
            # Another way to say this is:
            # If either the reward discrepancy is very small
            # or the sequence activity is very small, there is no change.
            #if trace_decay[t] > 0.:
            node_reward[i] += ((reward - node_reward[i]) *
                               trace_history[i, t_history[t]] *
                               mod_reward_rate *
                               trace_decay[t])

        # Update the trace history.
        parent_activity = node_activities[parent_index[i]]
        if element_index[i] == goal_index:
            trace_history[i, trace_index + 1] = parent_activity
        else:
            trace_history[i, trace_index + 1] = 0.

    trace_index += 1
    if trace_index >= trace_history_length:
        trace_index -= trace_history_length

    return node_reward, trace_history, trace_index
