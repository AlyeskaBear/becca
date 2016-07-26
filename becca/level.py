"""
The Level class.
"""

from __future__ import print_function
import numpy as np
import time

import becca.core.node as node
from becca.core.ziptie import ZipTie


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
    debug : boolean

    input_activities : array of floats
        The activity levels of each of the level's inputs.
    input_surplus : array of floats
        The input activities that are not accounted for in the
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
    max_num_inputs : int
        The maximum number of inputs that this level can accept.
    max_num_sequences : int
        The maximum number of sequences that this level will create.
    max_num_elements : int
        The maximum number of elements that this level will create.
    num_sequences : int
        The number of sequences that this level has already created.

    Node-specific
    TODO: add node-specific parameters here
    """
    def __init__(self,
                 level_index,
                 max_num_inputs,
                 max_num_elements,
                 max_num_sequences):
        """
        Configure the ``Level``.

        Parameters
        ---------
        level_index : int
            See ``Level.level_index``.
        max_num_inputs : int
            See ``Level.max_num_inputs``.
        max_num_sequences : int
            See ``Level.max_num_sequences``.
        max_num_elements : int
            See ``Level.max_num_elements``.
        """
        self.level_index = level_index
        self.name = "_".join(["level", str(self.level_index)])
        self.debug = False

        self.max_num_inputs = max_num_inputs
        self.max_num_elements = max_num_elements
        self.max_num_bundles = max_num_elements - max_num_inputs
        self.max_num_sequences = max_num_sequences
        self.num_sequences = 0
        # This limit on number of nodes accounts for the fact that the
        # root node and all its first generation children
        # (sequences of length 1) are the only nodes
        # not assigned to sequences.
        self.max_num_nodes = self.max_num_sequences + self.max_num_elements + 1
        self.num_nodes = 1

        # Normalization constants
        self.input_max = np.zeros(self.max_num_inputs)
        self.input_max_grow_time = 1e2
        self.input_max_decay_time = self.input_max_grow_time * 1e2

        self.activity_decay_rate = 1. / (2. ** self.level_index)
        self.activity_threshold = .1
        #TODO document
        self.reward_trace_length = 10
        self.trace_history_length = self.reward_trace_length + 2
        self.decay = np.zeros(self.reward_trace_length)
        for t in range(self.reward_trace_length):
            # Use a hyperbolic decay. A node will be assigned
            # some responsibility for the reward, even if it was active
            # a while ago. The amount of responsibility is proportional
            # to 1/t, where t is the number of time steps since the
            # node was active.
            self.decay[t] = 1. / (t + 1.)

        # Create a circular (cylindrical) buffer for trace history.
        #self.reward_history = [0.] * (self.reward_trace_length + 1)
        #self.node_trace_history = ([np.zeros(self.max_num_nodes)] *
        #                              (self.reward_trace_length + 1))
        self.node_trace_history = np.zeros((self.max_num_nodes,
                                            self.trace_history_length))
        # Keep track of the current location within the buffer.
        self.trace_index = 0

        #self.node_element_goal_votes = np.zeros((self.max_num_nodes,
        #                                         self.max_num_elements))

        self.input_activities = np.zeros(self.max_num_inputs)
        self.element_activities = np.zeros(self.max_num_elements)
        self.sequence_activities = np.zeros(self.max_num_sequences)
        self.element_goal_votes = np.zeros(self.max_num_elements)
        self.responsible_nodes = -np.ones(self.max_num_elements, 'int32')
        self.element_goals = np.zeros(self.max_num_elements)
        self.input_goals = np.zeros(self.max_num_inputs)
        self.sequence_goals = np.zeros(self.max_num_sequences)

        # Initialize the ziptie for bundling inputs.
        self.ziptie = ZipTie(self.max_num_inputs, 
                             num_bundles=self.max_num_bundles,
                             level=self.level_index)

        # Initialize nodes and their related data structures.
        self.node_activities = np.zeros(self.max_num_nodes)
        self.node_prev_activity = np.zeros(self.max_num_nodes)
        self.node_activity_threshold = 1e-6
        self.node_activity_rate = self.activity_decay_rate
        self.node_cumulative_activities = 1e-3 * np.ones(self.max_num_nodes)
        self.node_attempts = np.zeros(self.max_num_nodes)
        #self.node_total_attempts = 1e-3 * np.ones(self.max_num_nodes)
        self.node_fulfillment = 1e-3 * np.ones(self.max_num_nodes)
        self.node_unfulfillment = 1e-3 * np.ones(self.max_num_nodes)
        #self.node_choosability = np.zeros(self.max_num_nodes)
        self.node_curiosity_rate = 1e-4
        self.node_curiosity = 1e-1 * np.ones(self.max_num_nodes)
        self.node_reward_rate = 1e-3
        self.node_reward = np.zeros(self.max_num_nodes)
        self.node_value_to_parent = np.zeros(self.max_num_nodes)
        self.node_element_index = -np.ones(self.max_num_nodes, 'int32')
        self.node_sequence_index = -np.ones(self.max_num_nodes, 'int32')
        self.node_sequence_length = -np.ones(self.max_num_nodes, 'int32')
        self.node_child_threshold = 3e3
        self.node_num_children = np.zeros(self.max_num_nodes, 'int32')
        # The node index of each child child
        self.node_child_indices = -np.ones((self.max_num_nodes,
                                            self.max_num_elements), 'int32')
        self.node_parent_index = -np.ones(self.max_num_nodes, 'int32')

        # Node 0 is the root.
        # Give it one child for each input.
        # Don't assign sequences to these nodes. They represent sequences
        # of length 1 which aren't interesting. They're already represented
        # in this level's element inputs.
        self.node_sequence_length[0] = 0
        for i_element in range(self.max_num_elements):
            new_node_index = self.num_nodes
            self.node_child_indices[0, self.node_num_children[0]] = (
                new_node_index)
            self.node_cumulative_activities[new_node_index] = 1.
            self.node_sequence_length[new_node_index] = 1
            self.node_element_index[new_node_index] = i_element
            self.node_parent_index[new_node_index] = 0
            self.node_num_children[0] += 1
            self.num_nodes += 1

            # Give each of these nodes a set of child nodes.
            # This corresponds to initializing all sequences of length 2.
            for j_element in range(self.max_num_elements):
                # Don't allow same-element sequences.
                if i_element != j_element:
                    child_node_index = self.num_nodes
                    self.node_child_indices[
                        new_node_index,
                        self.node_num_children[new_node_index]] = (
                        child_node_index)
                    self.node_cumulative_activities[child_node_index] = 1.
                    self.node_sequence_length[child_node_index] = 2
                    self.node_element_index[child_node_index] = j_element
                    self.node_parent_index[child_node_index] = new_node_index
                    self.node_num_children[new_node_index] += 1
                    self.num_nodes += 1

        self.last_num_nodes = self.num_nodes


    def step(self, new_inputs, reward, satisfaction):
        """
        Advance all the sequences in the level by one time step.

        Start by updating all the inputs. Then walk through all
        the sequences, updating and training.

        Parameters
        ----------
        new_inputs : array of floats
            The inputs collected by the brain for the current time step.
        reward : float
            The current reward value.
        satisfaction : float
            The brain's current state of satisfaction, which is calculated
            from the recent reward history. If it hasn't received much
            reward recently, it won't be very satisfied.
        """
        node_index = 0
        self.update_inputs(new_inputs)

        # Run the inputs through the ziptie to find bundle activities
        # and to learn how to bundle them.
        bundle_activities = self.ziptie.sparse_featurize(self.input_activities)

        self.element_activities = np.concatenate((self.input_activities.copy(),
                                                  bundle_activities))

        self.ziptie.learn()
        self.num_active_elements = self.max_num_inputs + self.ziptie.num_bundles

        self.responsible_nodes = -np.ones(self.max_num_elements, 'int32')
        self.element_goal_votes = np.zeros(self.max_num_elements)
        #self.node_element_goal_votes = np.zeros((self.max_num_nodes,
        #                                         self.max_num_elements))
        prev_parent_activity = 1.
        parent_activity = 1.
        self.num_nodes, self.num_sequences = (
            node.step(node_index, # node parameters
                      self.node_activities,
                      self.node_prev_activity,
                      self.node_activity_threshold,
                      self.node_activity_rate,
                      self.node_cumulative_activities,
                      self.node_attempts,
                      #self.node_total_attempts,
                      self.node_fulfillment,
                      self.node_unfulfillment,
                      #self.node_choosability,
                      self.node_curiosity_rate,
                      self.node_curiosity,
                      self.node_reward,
                      self.node_value_to_parent,
                      self.node_element_index,
                      self.node_sequence_index,
                      self.node_sequence_length,
                      self.node_child_threshold,
                      self.node_num_children,
                      self.node_child_indices,
                      self.node_parent_index,
                      #self.node_element_goal_votes,
                      self.element_activities,
                      self.num_active_elements,
                      parent_activity,
                      prev_parent_activity,
                      self.sequence_activities,
                      self.num_nodes,
                      self.num_sequences,
                      self.max_num_sequences,
                      self.max_num_nodes,
                      self.element_goal_votes,
                      self.responsible_nodes,
                      self.element_goals,
                      self.sequence_goals,
                      reward,
                      satisfaction))

        if self.num_nodes > self.last_num_nodes: 
            #for i in xrange(self.last_num_nodes, self.num_nodes):
            #    sequence_indices = [self.node_element_index[i]]
            #    temp_index = i
            #    while self.node_parent_index[temp_index] != 0:
            #        sequence_indices.append(self.node_element_index[
            #            self.node_parent_index[temp_index]])
            #        temp_index = self.node_parent_index[temp_index]
            #    #print('  added sequence', sequence_indices[::-1])
            self.last_num_nodes = self.num_nodes

        # Decide which element goals to select, based on
        # all the votes tallied up across nodes.
        #arousal = 1/4.
        self.element_goals = np.zeros(self.max_num_elements)
        self.input_goals = np.zeros(self.max_num_inputs)
        #goal_index = np.where(self.element_goal_votes ** (1. / arousal)  >
        #                   np.random.random_sample( self.max_num_inputs))[0]
        matches = np.where(self.element_goal_votes ==
                           np.max(self.element_goal_votes))[0]
        goal_index = matches[np.argmax(np.random.random_sample(matches.size))]
        responsible_node = self.responsible_nodes[goal_index]
        #goal_index = np.argmax(self.element_goal_votes)

        #if (self.element_goal_votes[goal_index] ** (1. / arousal) >
        #        np.random.random_sample()):
        self.element_goals[goal_index] = 1.

        if goal_index < self.max_num_inputs:
            self.input_goals[goal_index] = 1.
        else:
            # Project element goals down to input goals.
            # 7 ms
            self.input_goals = self.ziptie.get_index_projection(
                goal_index - self.max_num_inputs)
        #else:
        #    goal_index = -1

        if self.debug:
            print('element goal votes', self.element_goal_votes)
            print('element goals', self.element_goals)
            print('input goals', self.input_goals)
            self.print_node(responsible_node)

        # Maintain the node activity history.
        #self.node_trace_history.pop(0)

        #self.node_trace_history.append(
        #    self.node_element_goal_votes[:, goal_index])

        #self.node_trace_history.append(self.node_activities.copy())
        #self.reward_history.append(reward)
        
        self.node_reward, self.node_trace_history, self.trace_index = (
            node.update_rewards(self.node_reward,
                                reward,
                                self.node_reward_rate,
                                self.reward_trace_length,
                                self.decay,
                                self.node_trace_history,
                                self.trace_history_length,
                                self.trace_index,
                                self.node_cumulative_activities,
                                self.node_activities,
                                self.node_element_index,
                                self.node_parent_index,
                                goal_index,
                                self.num_nodes
                                ))

        return self.sequence_activities


    def update_inputs(self, inputs, start_index=0):
        """
        Normalize and update inputs.

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

        After normalization, update each input activity with either
        a) the normalized value of its corresponding input or
        b) the decayed value, carried over from the previous time step,
        whichever is greater.

        Parameters
        ----------
        inputs : array of floats
            The current activity of the inputs.
        start_index : int
            The first input to update.

        Returns
        -------
        None
            self.input_activities is modified to include
            the normalized values of each of the inputs.
        """
        epsilon = 1e-8

        if start_index + inputs.size > self.max_num_inputs:
            print("level.Level.update_inputs:")
            print("    Attempting to update out of range input activities.")

        stop_index = min(start_index + inputs.size, self.max_num_inputs)
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

            self.input_activities[i] = max(val,
                                           self.input_activities[i] *
                                           (1. - self.activity_decay_rate))
            j += 1


    def visualize(self):
        """
        Show the current state of the ``Level``.
        """
        print(self.name)

        print("Input activities")
        for i_input, activity in enumerate(self.input_activities):
            if activity > self.activity_threshold:
                print(" ".join(["input", str(i_input), ":",
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
        # i, j, k, l are the input indices of each sequence
        # First, descend all the trees.
        def descend(node_index,
                    #prefix,
                    #sequence_indices,
                    #sequence_lists,
                    sequence_nodes):
            """
            Recursively descend the node trees and enumerate the sequences.
            """
            # If the current node index has a corresponding sequence,
            # add it to the list.
            #sequence_indices.append(self.node_sequence_index[node_index])
            #sequence_lists.append(prefix)
            sequence_nodes.append(node_index)

            # If this is a terminal node, backtrack up the tree.
            if self.node_num_children[node_index] == 0:
                return

            # If this isn't a terminal node, descend through all the
            # child children.
            for child_index in self.node_child_indices[
                    node_index, :self.node_num_children[node_index]]:
                if (self.node_element_index[child_index] <
                    self.num_active_elements):
                    descend(child_index, sequence_nodes)

        root_index = 0
        sequence_nodes = []
        descend(root_index,
                sequence_nodes)
        for i in sequence_nodes:
            self.print_node(i)
        print("==============================================================")
        

    def print_node(self, i):
        """ 
        Print a bunch of information about a node.
        """
        print("----------------------------------------------")

        full_sequence = [self.node_element_index[i]]
        temp_index = i
        while self.node_parent_index[temp_index] > 0:
            full_sequence.append(self.node_element_index[
                self.node_parent_index[temp_index]])
            temp_index = self.node_parent_index[temp_index]
        full_sequence = full_sequence[::-1]
        sequence_index = self.node_sequence_index[i] 
        if sequence_index == -1:
            sequence_index = "none"
        print("  node", i, 
              "  sequence", sequence_index,
              ": ", full_sequence)

        #Show the transitions within the sequence.
        print("    cumulative: {0:.4f}".format(
            self.node_cumulative_activities[i]))
        #print("      fulfillment: {0:.4f}".format(self.node_fulfillment[i]))
        #print("      unfulfillment: {0:.4f}".format(
        #    self.node_unfulfillment[i]))
        #print("      choosability: {0:.4f}".format(
        #    self.node_choosability[i]))
        print("    curiosity: {0:.4f}".format(self.node_curiosity[i]))
        print("    reward: {0:.4f}".format(self.node_reward[i]))
        #print("    val to parent: {0:.4f}".format(
        #    self.node_value_to_parent[i]))
        #total_goal_value = (self.node_curiosity[i] +
        #                    self.node_choosability[i] *
        #                    self.node_value_to_parent[i] / .9)
        #total_goal_value = min(max(total_goal_value, 0.), 1.)
        total_goal_value = self.node_reward[i] + self.node_curiosity[i]
        print("    total goal value: {0:.4f}".format(total_goal_value))
