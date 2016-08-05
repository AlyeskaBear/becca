"""
The Level class.
"""

from __future__ import print_function
import numpy as np

import becca.node as node
from becca.ziptie import Ziptie


class Level(object):
    """
    One level in the hierarchy of sequences.

    Each level contains a tree, representing all element sequences of
    length two. The nodes of the tree are an important data structure
    in level-related calculations. Their properties are listed in the
    Level.node_xxx attributes initialized below.

    In each time step, each level takes in new inputs, transforms them
    into sequences, and passes them up to the next level. The level also
    takes sequence goals that are passed down from above, transforms them
    into input goals, and passses them down to the next level. Finally,
    the level uses the new inputs to incrementally train itself for
    future time steps.
    """
    def __init__(self,
                 level_index,
                 max_num_inputs,
                 max_num_elements=None,
                 max_num_sequences=None
                ):
        """
        Configure the level.

        Parameters
        ---------
        level_index : int
            See Level.level_index.
        max_num_inputs : int
            See Level.max_num_inputs.
        max_num_sequences : int
            See Level.max_num_sequences.
        max_num_elements : int
            See Level.max_num_elements.
        """
        # level_index : int
        #     The index of this level.
        #     0 is the first and bottom level, and it counts up from there.
        self.level_index = level_index
        # name : str
        #     The name of this level.
        self.name = "_".join(["level", str(self.level_index)])
        # debug : boolean
        #     Print out extra information about the level's operation.
        self.debug = False

        # max_num_inputs : int
        # max_num_bundles : int
        # max_num_elements : int
        # max_num_nodes : int
        # max_num_sequences : int
        #     The maximum numbers of inputs, bundles, nodes, elements and
        #     sequences and that this level can accept.
        #     max_num_elements = max_num_inputs + max_num_bundles
        self.max_num_inputs = max_num_inputs
        self.max_num_bundles = self.max_num_inputs
        if max_num_elements is None:
            self.max_num_elements = self.max_num_inputs + self.max_num_bundles
        else:
            self.max_num_elements = max_num_elements
        if max_num_sequences is None:
            self.max_num_sequences = self.max_num_inputs
        else:
            self.max_num_sequences = max_num_sequences
        # This limit on the number of nodes accounts for
        # 1 root node, max_num_elements first generation nodes,
        # and each of those has children of all the other elements,
        # max_num_elements - 1.
        # This totals to:
        # 1 + max_num_elements + max_num_elements * (max_num_elements - 1)
        #     = 1 max_num_elements ** 2
        self.max_num_nodes = self.max_num_elements ** 2 + 1
        # num_sequences, num_nodes : int
        #     The number of nodes and sequences that this level
        #     has already created.
        #     num_sequences < num_nodes
        self.num_nodes = 1
        self.num_sequences = 0
        # num_active_elements : int
        #     The number of elements that have exhibited some non-trivial
        #     activity. Initially, the elements corresponding to bundles
        #     will be unused. Until they are populated, it's computationally
        #     more efficient to skip over them.
        self.num_active_elements = self.max_num_inputs
        # Normalization constants.
        # input_max : array of floats
        #     The maximum of each input's activity.
        #     Start with the assumption that each input has a
        #     maximum value of zero. Then adapt up from there
        #     based on the incoming observations.
        # input_max_decay_time, input_max_grow_time : float
        #     The time constant over which maximum estimates are
        #     decreased/increased. Growing time being lower allows the
        #     estimate to be weighted toward the maximum value.
        #self.input_max = np.zeros(self.max_num_inputs)
        self.input_max = np.zeros(self.max_num_inputs)
        self.input_max_grow_time = 1e2
        self.input_max_decay_time = self.input_max_grow_time * 1e2

        # activity_decay_rate : float
        #     The rate at which element activity decays between time steps.
        #     Decay takes five times longer for each additional level.
        self.activity_decay_rate = 1. / (5. ** self.level_index)

        # reward_trace_length : int
        #     The maximum number of time steps between when an action
        #     occurs and when it can claim responsibility for some of
        #     the current reward.
        self.reward_trace_length = 10
        # trace_history_length : int
        #     The length of the action history we need to save
        #     in order to calculate the reward trace.
        #     The extra 2 is due to the implementation of
        #     reward trace calculation.
        self.trace_history_length = self.reward_trace_length + 2
        # decay : array of floats
        #     The discount applied to a reward, depending on how many
        #     time steps after the action it occurred.
        #     if the action was taken at time t,
        #     decay[i] = discount for reward at time step t + 1 + i.
        self.decay = np.zeros(self.reward_trace_length)
        for time in range(self.reward_trace_length):
            # Use a hyperbolic decay.
            # The amount of credit is proportional
            # to 1/t, where t is the number of time steps since the
            # action was taken.
            self.decay[time] = 1. / (time  + 1.)

        # node_trace_history : 2D array of floats
        #     The trace history records, for each time step, all of the
        #     activities of all the nodes that voted for the action that
        #     was taken. These are nodes than can claim some responsibility
        #     for that action, and therefore nodes than can claim some
        #     credit for any subsequent reward. The data structure is
        #     a circular (cylindrical) buffer. Each row in the 2D array
        #     represents a single node. Each column represents a time step.
        #     As time marches forward, node responsibilties are written
        #     in columns to the right. After the last column is written,
        #     the history wraps around, and the next time step is written
        #     onto the first column again. Reading history works the same
        #     way, but working from right to left, wrapping around from
        #     first to last when necessary.
        self.node_trace_history = np.zeros((self.max_num_nodes,
                                            self.trace_history_length))
        # trace_index : int
        #     This is the index of the column in node_trace_history that
        #     represents the current time. It keeps track of
        #     the current location within the buffer.
        self.trace_index = 0

        # input_activities,
        # element_activities,
        # sequence_activitiies : array of floats
        #     inputs, elements, and sequences are all characterized by their
        #     activity--their level of activation at each time step.
        #     Activity for each input, element and sequence
        #     can vary between zero and one. Activity passes upward through
        #     a level. Input activities result in element activities,
        #     which result in sequence activities, which are then passed
        #     up to the next level as its input activities.
        self.input_activities = np.zeros(self.max_num_inputs)
        self.element_activities = np.zeros(self.max_num_elements)
        self.sequence_activities = np.zeros(self.max_num_sequences)

        # input_goals,
        # element_goals,
        # sequence_goals : array of floats
        #     Goals can be set for inputs, elements and sequences.
        #     They are temporary incentives, used for planning and
        #     action selection. They are passed down through the
        #     hierarchy, from higher levels to lower.
        #     Sequence goals are passed down from
        #     the level above. Nodes use those to set element goals.
        #     Element goals are then translated into input goals.
        #     Input goals are passed down to the next level as its
        #     new sequence goals. When the bottom level is reached,
        #     input goals corresponding to actions are used for
        #     action selection.
        self.input_goals = np.zeros(self.max_num_inputs)
        self.element_goals = np.zeros(self.max_num_elements)
        self.sequence_goals = np.zeros(self.max_num_sequences)

        # element_goal_votes : array of floats
        #     Each node votes for an element to become a goal.
        #     Each element keeps the highest vote it receives
        #     in each time step. element_goal_votes tallies
        #     these votes across nodes, and is then reset to
        #     zero at the beginning of eah time step.
        self.element_goal_votes = np.zeros(self.max_num_elements)
        # responsible_nodes : array of ints
        #     This tracks the index of the node that cast the
        #     highest vote for each element. That way, when an
        #     element is selected as a goal, we can see which
        #     node made it happen. This gives the mechanism
        #     some explainability which helps in visualization
        #     and debugging.
        self.responsible_nodes = -np.ones(self.max_num_elements, 'int32')

        # ziptie : Ziptie
        #     The ziptie is an instance of the Ziptie algorithm class,
        #     an incremental method for bundling inputs. Check out
        #     ziptie.py for a complete description. Zipties note which
        #     inputs tend to be co-active and creates bundles of them.
        #     This feature creation mechanism results in l0-sparse
        #     features, which sparsity helps keep Becca fast.
        self.ziptie = Ziptie(self.max_num_inputs,
                             num_bundles=self.max_num_bundles,
                             level=self.level_index)

        # Nodes are the data structure used to represent time.
        # They are arranged into a tree with a single root node,
        # child nodes for each element, and grandchild node for
        # each of the other elements. Each leaf in this tree
        # represents a two-step sequence. Nodes track several
        # aspects of the sequences.

        # node_activities, node_prev_activities : array of floats
        #     The activity of each node at both the current
        #     and the previous time step.
        self.node_activities = np.zeros(self.max_num_nodes)
        self.node_prev_activities = np.zeros(self.max_num_nodes)
        # node_activity_rate : float
        #     The rate at which node activities decay at each time
        #     step in the absence of any additional activation.
        #     new_activity = old_activity * (1 - node_acctivity_rate)
        self.node_activity_rate = self.activity_decay_rate
        # node_cumulative_activities : array of floats
        #     The sum of each node's activity over its lifetime.
        self.node_cumulative_activities = 1e-3 * np.ones(self.max_num_nodes)

        # node_attempts, node_fulfillment, node_unfulfillment : array of floats
        #     An attempt occurs when a node's parent is active and its
        #     terminal element is selected as a goal.
        #     All attempts result in either fulfillment or unfulfillment.
        #     If the node becomes active soon after the attempt, the attempt
        #     is considered fulfilled. The sooner and the larger the activity,
        #     the more complete the fulfillment. All attempts or portions of
        #     attempts that aren't fulfilled get added to unfulfillment.
        self.node_attempts = np.zeros(self.max_num_nodes)
        self.node_fulfillment = 1e-3 * np.ones(self.max_num_nodes)
        self.node_unfulfillment = 1e-3 * np.ones(self.max_num_nodes)

        # node_reward : array of floats
        #     The expected reward of choosing this node's element as a goal
        #     when its parent node is active.
        self.node_reward = np.zeros(self.max_num_nodes)
        # node_reward_rate : float
        #     A constant scaling factor on the rate at which node_reward adapts
        #     to new observations.
        self.node_reward_rate = 3e-3
        # node_curiosity : array of floats
        #     The value associated with selecting a node's element as a goal
        #     when the node's parent is active.
        self.node_curiosity = 1e-1 * np.ones(self.max_num_nodes)
        # node_curiosity_rate : float
        #     A constant scaling factor on the rate at which
        #     curiosity accumulates.
        #     A higher rate yields a more curious and exploratory brain.
        self.node_curiosity_rate = 1e-3

        # node_element_index, node_sequence_index : array of ints
        #     The indices associated with each node's terminal element
        #     and sequence, respectively. Note that nodes initially are
        #     not assigned to sequences. There are far fewer sequences
        #     than nodes and only the most commonly occurring nodes
        #     get elevated to the status of sequence.
        self.node_element_index = -np.ones(self.max_num_nodes, 'int32')
        self.node_sequence_index = -np.ones(self.max_num_nodes, 'int32')
        # node_sequence_threshold : float
        #     If a node's cumulative activity exceeds the
        #     node_sequence_threshold, then the node is elevated to the status
        #     of a sequence, an ouput from one level that becomes an input
        #     to the level above it.
        self.node_sequence_threshold = 1e2 * 4. ** self.level_index
        # node_sequence_length : array of ints
        #     The length of the sequence represented by a node.
        #     For root node, this is 0. For its children, 1.
        #     For the grandchild leaf nodes, it is 2.
        self.node_sequence_length = -np.ones(self.max_num_nodes, 'int32')

        # node_num_children : array of ints
        #     The number of child nodes that belong to each node.
        self.node_num_children = np.zeros(self.max_num_nodes, 'int32')
        # node_child_indices : 2D array of ints
        #     The node indices of the children that belong to each node.
        #     There should be as many non-negative child indices
        #     as their are num_children for each node.
        self.node_child_indices = -np.ones((self.max_num_nodes,
                                            self.max_num_elements), 'int32')
        # node_parent_index : array of ints
        #     The node index of each node's parent node.
        self.node_parent_index = -np.ones(self.max_num_nodes, 'int32')

        # Initialize the node tree.
        # Node 0 is the root. Give it one child for each element.
        # Don't assign sequences to these nodes. They represent sequences
        # of length 1 which aren't interesting. They're already represented
        # in this level's elements.
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
        # and to learn how to bundle them. nonbundle_activities
        # are all the input activities that don't contribute to
        # bundle_activites. It's what's left over. This forces all the
        # inputs to be expressed as concisely as possible,
        # in bundles wherever possible, rather than separately.
        (nonbundle_activities, bundle_activities) = (
            self.ziptie.featurize(self.input_activities))
        # The element activities are the combination of the residual
        # input activities and the bundle activities.
        self.element_activities = np.concatenate((nonbundle_activities,
                                                  bundle_activities))
        # Incrementally update the bundles in the ziptie.
        self.ziptie.learn()
        self.num_active_elements = self.max_num_inputs + self.ziptie.num_bundles

        # Re-initialize a few quantities for this time step.
        self.responsible_nodes = -np.ones(self.max_num_elements, 'int32')
        self.element_goal_votes = np.zeros(self.max_num_elements)
        prev_parent_activity = 1.
        parent_activity = 1.
        # Kick off the descent down through the node tree.
        self.num_nodes, self.num_sequences = (
            node.step(node_index, # node parameters
                      self.node_activities,
                      self.node_prev_activities,
                      self.node_activity_rate,
                      self.node_cumulative_activities,
                      self.node_attempts,
                      self.node_fulfillment,
                      self.node_unfulfillment,
                      self.node_curiosity_rate,
                      self.node_curiosity,
                      self.node_reward,
                      self.node_element_index,
                      self.node_sequence_index,
                      self.node_sequence_length,
                      self.node_sequence_threshold,
                      self.node_num_children,
                      self.node_child_indices,
                      self.node_parent_index,
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

        # If any nodes were added, reflect that.
        if self.num_nodes > self.last_num_nodes:
            # Print any element sequences that have been added this time step.
            if self.debug:
                for i in xrange(self.last_num_nodes, self.num_nodes):
                    sequence_indices = [self.node_element_index[i]]
                    temp_index = i
                    while self.node_parent_index[temp_index] != 0:
                        sequence_indices.append(self.node_element_index[
                            self.node_parent_index[temp_index]])
                        temp_index = self.node_parent_index[temp_index]
                    print('  added sequence', sequence_indices[::-1])
            self.last_num_nodes = self.num_nodes

        # Decide which element goals to select, based on
        # all the votes tallied up across nodes.
        # Choose exactly one goal at each time step.
        # Choose the element with the largest vote. If there is a tie,
        # randomly select between them.
        self.element_goals = np.zeros(self.max_num_elements)
        self.input_goals = np.zeros(self.max_num_inputs)
        matches = np.where(self.element_goal_votes ==
                           np.max(self.element_goal_votes))[0]
        goal_index = matches[np.argmax(np.random.random_sample(matches.size))]
        # Track which node cast the winning vote.
        responsible_node = self.responsible_nodes[goal_index]
        self.element_goals[goal_index] = 1.
        # Express the element goal in terms of input goals.
        if goal_index < self.max_num_inputs:
            self.input_goals[goal_index] = 1.
        else:
            # If the goal element is a bundle, cast it in terms of inputs.
            # Project element goals down to input goals.
            self.input_goals = self.ziptie.get_index_projection(
                goal_index - self.max_num_inputs)

        if self.debug:
            print('element goal votes', self.element_goal_votes)
            print('element goals', self.element_goals)
            print('input goals', self.input_goals)
            self.print_node(responsible_node)

        # Assign credit for the current reward to any goals set
        # and actions taken in the last few time steps.
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
        # If any of the nodes are assigned to this level's
        # sequences, their activities will be reflected
        # in the sequence activities and pushed upward to be
        # inputs for the next level.
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

        # This is written to be easily compilable by numba, however,
        # it hasn't proven to be at all significant in profiling, so it
        # has stayed in slow-loop python for now.
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
            # Ensure that 0 <= val <= 1.
            val = max(0., val)
            val = min(1., val)

            self.input_activities[i] = max(val,
                                           self.input_activities[i] *
                                           (1. - self.activity_decay_rate))
            j += 1


    def visualize(self):
        """
        Show the current state of the level.
        """
        # activity_threshold the level at which we can ignore
        # an element's activity in order to simplify display.
        activity_threshold = .01
        print(self.name)

        print("Input activities")
        for i_input, activity in enumerate(self.input_activities):
            if activity > activity_threshold:
                print(" ".join(["input", str(i_input), ":",
                                "activity ", str(activity)]))

        print("Sequence activities")
        for i_sequence, activity in enumerate(self.sequence_activities):
            if activity > activity_threshold:
                print(" ".join(["sequence", str(i_sequence), ":",
                                "activity ", str(activity)]))

        # Enumerate all sequences, doing a depth-first descent
        # through the tree.
        def descend(node_index,
                    sequence_nodes):
            """
            Recursively descend the node trees and enumerate the sequences.
            """
            # If the current node index has a corresponding sequence,
            # add it to the list.
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

        # Re-create the node's sequence by tracing its parentage.
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
        print("    curiosity: {0:.4f}".format(self.node_curiosity[i]))
        print("    reward: {0:.4f}".format(self.node_reward[i]))
        total_goal_value = self.node_reward[i] + self.node_curiosity[i]
        print("    total goal value: {0:.4f}".format(total_goal_value))
