"""
The Level class.
"""

from __future__ import print_function
import numpy as np

import becca.core.node as node
import becca.core.ziptie as ziptie


class Level(object):
    """
    One level in the hierarchy of sequences.

    Attributes
    ----------
    activity_threshold : float
        Threshold below which input activity is treated as zero.
    activity_decay_rate : float
        The fraction that sequence activities decay.
        0 <= decay_rate <= 1.
    element_activities : array of floats
        The activity levels of each of the level's elements.
    element_surplus : array of floats
        The element activities that are not accounted for in the
        currently active sequences.
    goal_decay_rate : float
        The fraction that each goal will decay during each time step.
    input_max : array of floats
        The maximum of each input's activity.
    input_max_decay_time, input_max_grow_time : float
        The time constant over which maximum estimates are
        decreased/increased. Growing time being lower allows the
        estimate to be weighted toward the maximum value.
    level_index : int
        The index of this instance of ``Level``.
        0 is the first and bottom ``Level``, and it counts up from there.
    name : str
        The name of this ``Level``.
    max_num_elements : int
        The maximum number of elements that this level can accept.
    max_num_sequences : int
        The maximum number of sequences that this level will create.
    max_num_sets : int
        The maximum number of sets that this level will create.
    num_sequences : int
        The number of sequences that this level has already created.

    Node-specific
    TODO: add node-specific parameters here
    """
    def __init__(self,
                 level_index,
                 max_num_elements,
                 max_num_sets,
                 max_num_sequences):
        """
        Configure the ``Level``.

        Parameters
        ---------
        level_index : int
            See ``Level.level_index``.
        max_num_elements : int
            See ``Level.max_num_elements``.
        max_num_sequences : int
            See ``Level.max_num_sequences``.
        max_num_sets : int
            See ``Level.max_num_sets``.
        """
        self.level_index = level_index
        self.name = "_".join(["level", str(self.level_index)])

        self.max_num_elements = max_num_elements
        self.max_num_sets = self.max_num_sets
        self.max_num_bundles = max_num_sets - max_num_elements
        self.max_num_sequences = max_num_sequences
        self.num_sequences = 0
        # This limit on number of nodes accounts for the fact that the
        # root node and all its first generation children
        # (sequences of length 1) are the only nodes
        # not assigned to sequences.
        #self.max_num_nodes = self.max_num_sequences + self.max_num_elements + 1
        self.max_num_nodes = self.max_num_sequences + self.max_num_sets + 1
        self.num_nodes = 1

        # Normalization constants
        self.input_max = np.zeros(self.max_num_elements)
        self.input_max_grow_time = 1e2
        self.input_max_decay_time = self.input_max_grow_time * 1e2

        self.activity_decay_rate = 1. / (2. ** (self.level_index + 0.))
        self.activity_threshold = .1
        #self.goal_decay_rate = self.activity_decay_rate / 2.
        #TODO document
        self.reward_trace_length = 10
        self.node_activity_history = ([np.zeros(self.max_num_nodes)] *
                                      self.reward_trace_length)

        self.element_activities = np.zeros(self.max_num_elements)
        self.sequence_activities = np.zeros(self.max_num_sequences)
        self.goal_votes = np.zeros(self.max_num_elements)
        self.element_goals = np.zeros(self.max_num_elements)
        self.sequence_goals = np.zeros(self.max_num_sequences)

        # Initialize the ziptie for bundling elements.
        self.ziptie = Ziptie(self.max_num_elements, level=self.level_index)

        # Initialize nodes and their related data structures.
        self.node_activities = np.zeros(self.max_num_nodes)
        self.node_prev_activity = np.zeros(self.max_num_nodes)
        self.node_activity_threshold = 1e-6
        self.node_activity_rate = self.activity_decay_rate
        self.node_cumulative_activity = 1e-3 * np.ones(self.max_num_nodes)
        self.node_attempts = np.zeros(self.max_num_nodes)
        self.node_total_attempts = 1e-3 * np.ones(self.max_num_nodes)
        self.node_fulfillment = 1e-3 * np.ones(self.max_num_nodes)
        self.node_unfulfillment = np.ones(self.max_num_nodes)
        self.node_choosability = np.zeros(self.max_num_nodes)
        self.node_curiosity_rate = 1e-3
        self.node_curiosity = 1e-1 * np.ones(self.max_num_nodes)
        self.node_reward_rate = 1e-2
        self.node_reward = np.zeros(self.max_num_nodes)
        self.node_value_to_parent = np.zeros(self.max_num_nodes)
        #self.node_value_rate = self.activity_decay_rate
        self.node_element_index = -np.ones(self.max_num_nodes, 'int32')
        self.node_sequence_index = -np.ones(self.max_num_nodes, 'int32')
        #self.node_buds = np.zeros((self.max_num_nodes,
        #                           self.max_num_elements))
        self.node_child_threshold = 1e2
        self.node_num_children = np.zeros(self.max_num_nodes, 'int32')
        # The node index of each child child
        self.node_child_indices = -np.ones((self.max_num_nodes,
                                             self.max_num_elements), 'int32')
        # Node 0 is the root.
        # Give it one child for each element.
        # Don't assign sequences to these nodes. They represent sequences
        # of length 1 which aren't interesting. They're already represented
        # in this level's input elements.
        for i_element in range(self.max_num_elements):
            node_index = self.num_nodes
            self.node_child_indices[0, i_element] = node_index
            self.node_element_index[node_index] = i_element
            self.node_num_children[0] += 1
            self.num_nodes += 1


    def step(self, inputs, reward, satisfaction):
        """
        Advance all the sequences in the level by one time step.

        Start by updating all the elements. Then walk through all
        the sequences, updating and training.

        Parameters
        ----------
        inputs : array of floats
            The inputs collected by the brain for the current time step.
        reward : float
            The current reward value.
        satisfaction : float
            The brain's current state of satisfaction, which is calculated
            from the recent reward history. If it hasn't received much
            reward recently, it won't be very satisfied.
        """
        node_index = 0
        self.update_elements(inputs)
        self.goal_votes = np.zeros(self.max_num_elements)
        prev_parent_activity = 1.
        parent_activity = 1.
        self.num_nodes, self.num_sequences = (
            node.step(node_index, # node parameters
                      self.node_activities,
                      self.node_prev_activity,
                      self.node_activity_threshold,
                      self.node_activity_rate,
                      self.node_cumulative_activity,
                      self.node_attempts,
                      self.node_total_attempts,
                      self.node_fulfillment,
                      self.node_unfulfillment,
                      self.node_choosability,
                      self.node_curiosity_rate,
                      self.node_curiosity,
                      #self.node_reward_rate,
                      self.node_reward,
                      self.node_value_to_parent,
                      #self.node_value_rate,
                      self.node_element_index,
                      self.node_sequence_index,
                      #self.node_buds,
                      self.node_child_threshold,
                      self.node_num_children,
                      self.node_child_indices,
                      self.element_activities, # level parameters
                      parent_activity,
                      prev_parent_activity,
                      self.sequence_activities,
                      self.num_nodes,
                      self.num_sequences,
                      self.max_num_sequences,
                      self.goal_votes,
                      self.element_goals,
                      #element_total_weights,
                      #element_weighted_values,
                      #min_element_values,
                      self.sequence_goals,
                      reward,
                      satisfaction))

        # Maintain the node activity history.
        self.node_activity_history.pop()
        self.node_activity_history.append(self.node_activities.copy())

        # Update node reward estimates.
        #
        # Adjust the reward update rate so that the node adjusts
        # very quickly to the first few exposures and then gradually
        # more slowly to subsequent ones. This tends to decrease training
        # time. It's an agressive learning strategy, and so is
        # prone to initial error, but it allows
        # for later experiences to refine the reward estimate and
        # eventually correct that error out.
        mod_reward_rate = np.maximum(self.node_reward_rate,
                                     1. / (2. + self.node_cumulative_activity))
        for i in range(len(self.node_activity_history)):
            # Use a hyperbolic decay. A node will be assigned
            # some responsibility for the reward, even if it was active
            # a while ago. The amount of responsibility is proportional
            # to 1/t, where t is the number of time steps since the
            # node was active.
            decay = 1. / (i + 1.)

            # Cycle through the node activity history, starting with the
            # most recent time step and working backward.
            i_history = len(self.node_activity_history) - i - 1

            # Increment the expected reward value associated with each sequence.
            # The size of the increment is larger when:
            #     1. the discrepancy between the previously learned and
            #         observed reward values is larger and
            #     2. the sequence activity is greater.
            # Another way to say this is:
            # If either the reward discrepancy is very small
            # or the sequence activity is very small, there is no change.
            self.node_reward += ((reward - self.node_reward) *
                                 self.node_activity_history[i_history] *
                                 mod_reward_rate * decay)

        # Decide which element goals to set, based on
        # all the votes tallied up across nodes.
        arousal = 1.
        self.element_goals = np.zeros(self.max_num_elements)
        #i_goals = np.where(self.goal_votes ** (1. / arousal)  > 
        #                   np.random.random_sample( self.max_num_elements))[0]
        i_goals = np.argmax(self.goal_votes)
        if (self.goal_votes[i_goals] ** (1. / arousal)  > 
            np.random.random_sample()):
            self.element_goals[i_goals] = 1.

        #print('goal votes', self.goal_votes)
        #print('element goals', self.element_goals)

        return self.sequence_activities


    def update_elements(self, inputs, start_index=0):
        """
        Normalize and update elements.

        Normalize activities so that they are predictably distrbuted.
        Use a running estimate of the maximum of each cable activity.
        Scale it so that the max would fall at 1.

        Normalization has several benefits.
        1. It makes for fewer constraints on worlds and sensors.
           It allows any sensor can return any range of values.
        2. Gradual changes in sensors and the world can be adapted to.
        3. It makes the bundle creation heuristic more robust and
           predictable. The approximate distribution of cable
           activities is known and can be designed for.

        After normalization, update each element activity with either
        a) the normalized value of its corresponding input or
        b) the decayed value, carried over from the previous time step,
        whichever is greater.

        Parameters
        ----------
        inputs : array of floats
            The current activity of the inputs.
        start_index : int
            The first element to update.

        Returns
        -------
        None
            self.element_activities is modified to include
            the normalized values of each of the elements.
        """
        epsilon = 1e-8

        if start_index + inputs.size > self.max_num_elements:
            print("level.Level.update_elements:")
            print("    Attempting to update out of range element activities.")

        stop_index = min(start_index + inputs.size, self.max_num_elements)
        # Input index
        j = 0
        for i in range(start_index, stop_index):
            val = inputs[j]

            # Decay the maximum value.
            self.input_max[i] += ((val - self.input_max[i]) /
                                  self.input_max_decay_time)

            # Grow the maximum value, when appropriate.
            if val > self.input_max[i]:
                self.input_max[i] += ((val - self.input_max[i]) /
                                      self.input_max_grow_time)

            # Scale the input by the maximum.
            val = val / (self.input_max[i] + epsilon)
            val = max(0., val)
            val = min(1., val)

            # Sparsify the cable activities to speed up processing.
            if val < self.activity_threshold:
                val = 0.

            self.element_activities[i] = max(val,
                                             self.element_activities[i] * 
                                             (1. - self.activity_decay_rate))
            j += 1


    def visualize(self):
        """
        Show the current state of the ``Level``.
        """
        print(self.name)

        print("Element_activities")
        for i_element, activity in enumerate(self.element_activities):
            if activity > self.activity_threshold:
                print(" ".join(["element", str(i_element), ":",
                                "activity ", str(activity)]))

        print("Sequence activities")
        for i_sequence, activity in enumerate(self.sequence_activities):
            if activity > self.activity_threshold:
                print(" ".join(["sequence", str(i_sequence), ":",
                                "activity ", str(activity)]))

        # Enumerate all sequences.
        # Use the format
        #   q: i - j - k - l
        # where q is the sequence index and
        # i, j, k, l are the element indices of each sequence
        # First, descend all the trees.
        def descend(node_index,
                    prefix,
                    sequence_indices,
                    sequence_lists,
                    sequence_nodes):
            """
            Recursively descend the node trees and enumerate the sequences.
            """
            # If the current node index has a corresponding sequence,
            # add it to the list.
            sequence_indices.append(self.node_sequence_index[node_index])
            sequence_lists.append(prefix)
            sequence_nodes.append(node_index)

            # If this is a terminal node, backtrack up the tree.
            if self.node_num_children[node_index] == 0:
                return

            # If this isn't a terminal node, descend through all the
            # child children.
            for child_index in self.node_child_indices[
                    node_index, :self.node_num_children[node_index]]:
                new_prefix = prefix + [self.node_element_index[child_index]]
                descend(child_index,
                        new_prefix,
                        sequence_indices,
                        sequence_lists,
                        sequence_nodes)

        root_index = 0
        prefix = []
        sequence_indices = []
        sequence_lists = []
        sequence_nodes = []
        descend(root_index,
                prefix,
                sequence_indices,
                sequence_lists,
                sequence_nodes)

        for i, seq_index in enumerate(sequence_indices):
            print("  sequence", seq_index, ': ', sequence_lists[i])
            j = sequence_nodes[i]

            #Show the transitions within the sequence.
            print("    cumulative: {0:.4f}".format(
                self.node_cumulative_activity[j]))
            print("      fulfillment: {0:.4f}".format(self.node_fulfillment[j]))
            print("      unfulfillment: {0:.4f}".format(
                self.node_unfulfillment[j]))
            print("      choosability: {0:.4f}".format(
                self.node_choosability[j]))
            print("    curiosity: {0:.4f}".format(self.node_curiosity[j]))
            print("    reward: {0:.4f}".format(self.node_reward[j]))
            print("    val to parent: {0:.4f}".format(
                self.node_value_to_parent[j]))
            total_goal_value = (self.node_curiosity[j] +
                                self.node_choosability[j] *
                                self.node_value_to_parent[j] / .9)
            total_goal_value = min(max(total_goal_value, 0.), 1.)
            print("    total goal value: {0:.4f}".format(total_goal_value))
            print("----------------------------------------------")
        print("==============================================================")
