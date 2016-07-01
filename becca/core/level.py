"""
The Level class.
"""

from __future__ import print_function
import numpy as np

import becca.core.node as node


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
        The maximum number of sequences that this level can create.
    num_sequences : int
        The number of sequences that this level has already created.

    Node-specific
    TODO: add node-specific parameters here
    """
    def __init__(self, level_index, max_num_elements, max_num_sequences):
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
        """
        self.level_index = level_index
        self.name = "_".join(["level", str(self.level_index)])

        self.max_num_elements = max_num_elements
        self.max_num_sequences = max_num_sequences
        self.num_sequences = 0
        # This limit on number of nodes accounts for the fact that the
        # root node and all its first generation branches
        # (sequences of length 1) are the only nodes
        # not assigned to sequences.
        self.max_num_nodes = self.max_num_sequences + self.max_num_elements + 1
        self.num_nodes = 1

        # Normalization constants
        self.input_max = np.zeros(self.max_num_elements)
        self.input_max_grow_time = 1e2
        self.input_max_decay_time = self.input_max_grow_time * 1e2

        self.activity_decay_rate = 1. / (2. ** self.level_index)
        self.activity_threshold = .1

        self.element_activities = np.zeros(self.max_num_elements)
        self.sequence_activities = np.zeros(self.max_num_sequences)
        self.element_goals = np.zeros(self.max_num_elements)
        self.sequence_goals = np.zeros(self.max_num_sequences)

        # Initialize nodes and their related data structures.
        self.node_activity = np.zeros(self.max_num_nodes)
        self.node_prev_activity = np.zeros(self.max_num_nodes)
        self.node_activity_threshold = 1e-2
        self.node_activity_rate = self.activity_decay_rate
        self.node_cumulative_activity = 1e-3 * np.ones(self.max_num_nodes)
        self.node_curiosity_rate = 1e-3
        self.node_curiosity = np.zeros(self.max_num_nodes)
        self.node_reward_rate = 1e-2
        self.node_reward = np.zeros(self.max_num_nodes)
        self.node_total_value = np.zeros(self.max_num_nodes)
        self.node_value_rate = self.activity_decay_rate
        self.node_element_index = -np.ones(self.max_num_nodes, 'int32')
        self.node_sequence_index = -np.ones(self.max_num_nodes, 'int32')
        self.node_buds = np.zeros((self.max_num_nodes,
                                   self.max_num_elements))
        self.node_bud_threshold = 1e1
        self.node_num_branches = np.zeros(self.max_num_nodes, 'int32')
        # The node index of each child branch
        self.node_branch_indices = -np.ones((self.max_num_nodes,
                                             self.max_num_elements), 'int32')
        # Node 0 is the root.
        # Give it one branch for each element.
        # Don't assign sequences to these nodes. They represent sequences
        # of length 1 which aren't interesting. They're already represented
        # in this level's input elements.
        for i_element in range(self.max_num_elements):
            node_index = self.num_nodes
            self.node_branch_indices[0, i_element] = node_index
            self.node_element_index[node_index] = i_element
            self.node_num_branches[0] += 1
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
        upstream_activity = 1.
        self.num_nodes, self.num_sequences = (
            node.step(node_index, # node parameters
                      self.node_activity,
                      self.node_prev_activity,
                      self.node_activity_threshold,
                      self.node_activity_rate,
                      self.node_cumulative_activity,
                      self.node_curiosity_rate,
                      self.node_curiosity,
                      self.node_reward_rate,
                      self.node_reward,
                      self.node_total_value,
                      self.node_value_rate,
                      self.node_element_index,
                      self.node_sequence_index,
                      self.node_buds,
                      self.node_bud_threshold,
                      self.node_num_branches,
                      self.node_branch_indices,
                      self.element_activities, # level parameters
                      upstream_activity,
                      self.sequence_activities,
                      self.num_nodes,
                      self.num_sequences,
                      self.max_num_sequences,
                      self.element_goals,
                      self.sequence_goals,
                      reward,
                      satisfaction))

        #for node in self.nodes:
        #    self.num_sequences = node.step(self.element_activities,
        #                                   upstream_activity,
        #                                   self.sequence_activities,
        #                                   self.num_sequences,
        #                                   self.max_num_sequences,
        #                                   self.element_goals,
        #                                   self.sequence_goals,
        #                                   reward,
        #                                   satisfaction)

        return self.sequence_activities


    #@jit(nopython=True)
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

        stop_index = np.minimum(start_index + inputs.size,
                                self.max_num_elements)
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
            val = np.maximum(0., val)
            val = np.minimum(1., val)

            # Sparsify the cable activities to speed up processing.
            if val < self.activity_threshold:
                val = 0.

            self.element_activities[i] = np.maximum(
                val,
                self.element_activities[i] * (1. - self.activity_decay_rate))
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
            if self.node_sequence_index[node_index] != -1:
                sequence_indices.append(self.node_sequence_index[node_index])
                sequence_lists.append(prefix)
                sequence_nodes.append(node_index)

            # If this is a terminal node, backtrack up the tree.
            if self.node_num_branches[node_index] == 0:
                return

            # If this isn't a terminal node, descend through all the
            # child branches.
            for branch_index in self.node_branch_indices[
                    node_index, :self.node_num_branches[node_index]]:
                new_prefix = prefix + [self.node_element_index[branch_index]]
                descend(branch_index,
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
            print("    curiosity: {0:.4f}".format(self.node_curiosity[j]))
            print("    reward: {0:.4f}".format(self.node_reward[j]))

            print("    buds")
            row_string = "        "
            for k, bud in enumerate(self.node_buds[j, :]):
                if bud > self.activity_threshold:
                    bud_string = "    {0}: {1:.2f}".format(k, bud)
                    row_string += bud_string
            print(row_string)

        print("======================================================")
