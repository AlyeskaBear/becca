"""
The Node class.
"""

import numpy as np
#import numba as nb
#import tools
#@numba.jit
class Node(object):
    """
    A sequence node. Sequences are represented as trees of these nodes.

    Attributes
    ----------
    activity : float
        The current activity of this Node.
    activity_rate : float
        The rate at which activities decay over time. A rate of 1 means
        that activities decay instantly. A rate of 0 means that they
        never decay.
    activity_threshold : float
        The value below which activity will be treated as zero. Doing this
        sparsifies the system and reduces the number of calculations
        required without having a large impact on the results.
    buds : array of floats
        The cumulative activities associated with each of the elements,
        as potential child nodes.
    bud_threshold : float
        The level of accumulated activity at which a bud gets turned
        into a full-blown child node.
    cumulative_activity : float
        The aggregated sum of past sequence activity.
    curiosity : float
        Similar to reward, but not based on experience. This is intended to
        represent an optimism that encourages exploration.
    curiosity_rate : float
        A scaling factor that influences the rate at which
        curiosity accumulates.
    element_activities : array of floats
        The most recent of observed element activities.
    element_index : array of ints
        The index of the element that is represented by this Node.
    nodes : array of Nodes
        The child nodes for this one. Each of these extend the sequence
        in a new direction.
    num_elements : int
        The total number of elements that this Node will see.
    previous_activity : array of floats
        The observed Node's activity from the previous time step.
    reward : float
        An estimate of the reward obtained when this sequence node is active.
    reward_rate : float
        The rate at which the reward is incrementally stepped toward
        the most recent observed value.
        0 <= reward_learning_rate <= 1.
        Smaller rates have a longer learning time, but maintain more
        statistically accurate estimates.
    sequence_index : int
        The index of the sequence that the node corresponds to.
    total_value : float
        The overall value associated with the sequence, a combination of
        reward, curiosity, top-down goals and goals associated with
        longer sequences, for which this is a prefix.
    value_rate : float
        The rate, between 0 and 1, at which the Node's total value decays.
        0 is no decay. 1 is instant decay.
    """

    def __init__(self, num_elements, element_index, sequence_index=None,
                 cumulative_activity=0., activity_rate=1.):
        """
        Set up the Node.

        Parameters
        ----------
        activity_rate : float
            See self.activity_rate.
        element_index : int
            See self.element_index.
        num_elements : int
            See self.num_elements.
        sequence_index : int
            See self.sequence_index
        """
        self.num_elements = num_elements
        self.activity = 0.
        self.previous_activity = 0.
        self.activity_threshold = 1e-2
        self.cumulative_activity = cumulative_activity
        self.activity_rate = activity_rate
        self.curiosity_rate = 1e-3
        self.curiosity = 0.
        self.reward_rate = 1e-2
        self.reward = 0.
        self.total_value = 0.
        self.value_rate = activity_rate
        self.element_index = element_index
        self.sequence_index = sequence_index
        self.element_activities = np.zeros(self.num_elements)
        self.buds = np.zeros(self.num_elements)
        self.bud_threshold = 1e3
        self.nodes = []

    def step(self, element_activities,
             upstream_activity, sequence_activities,
             num_sequences, max_num_sequences,
             element_goals, sequence_goals,
             reward, satisfaction):
        """
        Update the Node with the most recent element activities.

        Parameters
        ----------
        element_activities : array of floats
            The most recent of observed element activities.
        element_goals : array of floats
            The goals (internal reward due to planning and incentives)
            associated with each element.
        reward : float
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

        # Calculate the new node activity.
        # Node activity is either the incoming activity associated with
        # the element index, or the activity from the upstream node,
        # whichever is less.
        element_activity = element_activities[self.element_index]
        self.activity = np.minimum(upstream_activity, element_activity)
        # If the decayed activity from the previous tme step was greater,
        # revert to that.
        self.activity = np.maximum(self.previous_activity *
                                   (1. - self.activity_rate),
                                   self.activity)
        self.cumulative_activity += self.activity

        # Set the relevant sequence activity for the Level.
        if self.sequence_index is not None:
            sequence_activities[self.sequence_index] = self.activity

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
        self.reward += ((reward - self.reward) *
                        self.activity *
                        self.reward_rate)

        # Fulfill curiosity.
        self.curiosity *= 1. - self.activity
        # Update the curiosity.
        uncertainty = 1. / (1. + self.cumulative_activity)
        self.curiosity += (self.curiosity_rate *
                           uncertainty *
                           (1. - self.curiosity) *
                           (1. - satisfaction))

        # Calculate the forward goal, the goal value associated with
        # longer sequences, for which this is a prefix. A child node's
        # value propogates back to this node with a fraction of its
        # value, given by:
        #       fraction = child node's cumulative activity /
        #                   this node's cumulative activity
        # The forward goal is the maximum propogated goal across
        # all child nodes.
        forward_goal = -1
        for node in self.nodes:
            node_value = node.total_value * (node.cumulative_activity /
                                             self.cumulative_activity)
            if node_value > forward_goal:
                node_value = forward_goal

        # The goal value passed down from the level above.
        top_down_goal = sequence_goals[self.sequence_index]

        # The new element value is the maximum of all its constituents.
        total_value_ceil = np.maximum([forward_goal,
                                       top_down_goal,
                                       self.curiosity,
                                       self.reward])
        new_total_value = total_value_ceil * (1. - self.activity)
        self.total_value = np.maximum(new_total_value,
                                      self.total_value *
                                      (1. - self.value_rate))
        # Pass the new element goal down.
        element_goals[self.element_index] = self.total_value

        # Descend through each of the child branches.
        for node in self.nodes:
            node.step(element_activities,
                      self.previous_activity,
                      sequence_activities,
                      num_sequences,
                      max_num_sequences,
                      element_goals,
                      sequence_goals,
                      reward,
                      satisfaction)

        # If there is still room for more sequences, grow them.
        if num_sequences < max_num_sequences:
            # Update each of the child buds.
            self.buds += self.activity * self.element_activities
            # Don't allow the node to have a child with the same element index,
            self.buds[self.element_index] = 0.
            # or with nodes that they already have.
            for node in self.nodes:
                self.buds[node.element_index] = 0.

            # Create a new branch when appropriate.
            matches = np.where(self.buds > self.bud_threshold)[0]
            if matches.size > 0:
                element_index = matches[0]
                self.nodes.append(Node(self.activity_rate,
                                       element_index,
                                       sequence_index=num_sequences,
                                       cumulative_activity=self.bud_threshold))
                self.buds[element_index] = 0.

        # Pass the node activity to the next time step.
        self.previous_activity = self.activity


'''

    def visualize(self):
        """
        Show the transitions within the sequence.
        """
        print("      curiosity: {0:.4f}".format(self.curiosity))
        print("      reward: {0:.4f}".format(self.reward))

        print("      observations")
        for i in range(self.num_elements):
            row_string = "        "
            for j in range(self.num_elements):
                element_string = " {0:>12,.2f}".format(self.observations[i,j]) 
                row_string += element_string
            print(row_string)

        print("      opportunities")
        for i in range(self.num_elements):
            row_string = "        "
            for j in range(self.num_elements):
                element_string = " {0:>12,.2f}".format(self.opportunities[i,j]) 
                row_string += element_string
            print(row_string)

        print("      transition_strengths")
        for i in range(self.num_elements):
            row_string = "        "
            for j in range(self.num_elements):
                element_string = " {0:>12,.2f}".format(
                        self.transition_strengths[i,j]) 
                row_string += element_string
            print(row_string)

if __name__ == "__main__":
    test_sequence = Sequence(1.)
    test_sequence.visualize()
'''
