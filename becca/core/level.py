"""
The Level class.
"""

from __future__ import print_function
#import numba
import numpy as np

from becca.core.node import Node

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
        self.max_num_elements = max_num_elements
        self.max_num_sequences = max_num_sequences
        self.num_sequences = 0
        self.level_index = level_index
        self.name = "_".join(["level", str(self.level_index)])

        # Normalization constants
        self.input_max = np.zeros(self.max_num_elements)
        self.input_max_grow_time = 1e2
        self.input_max_decay_time = self.input_max_grow_time * 1e2

        self.activity_decay_rate = 1. - 1. / (2. ** self.level_index)
        self.reward_learning_rate = 1e-3
        self.activity_threshold = .1

        self.nodes = []
        for index in range(self.max_num_elements):
            self.nodes.append(Node(self.max_num_elements, index,
                                   activity_rate=self.activity_decay_rate))

        self.element_activities = np.zeros(self.max_num_elements)
        #self.last_element_activities = np.zeros(self.max_num_elements)
        #self.last_element_surplus = np.zeros(self.max_num_elements)
        self.sequence_activities = np.zeros(self.max_num_sequences)
        #self.last_sequence_activities = np.zeros(self.max_num_sequences)
        #self.sequence_rewards = np.zeros(self.max_num_sequences)
        self.element_goals = np.zeros(self.max_num_elements)
        self.sequence_goals = np.zeros(self.max_num_sequences)


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
        self.update_elements(inputs)
        upstream_activity = 1.
        for node in self.nodes:
            node.step(self.element_activities,
                      upstream_activity,
                      self.sequence_activities,
                      self.num_sequences,
                      self.max_num_sequences,
                      self.element_goals,
                      self.sequence_goals,
                      reward,
                      satisfaction)

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
                self.element_activities[i], val)
            j += 1


    def visualize(self):
        """
        Show the current state of the ``Level``.
        """
        print(self.name)

'''
        self.element_surplus = self.element_activities.copy()

        # Calculate and return the activities of each of the sequences.
        for i_sequence in np.arange(self.ziptie.num_bundles - 1, -1, -1):

            # Get the set of elements that contribute to the sequence.
            i_elements = self.ziptie.get_elements(i_sequence)

            # Calculate the activity of each sequence.
            new_activity = self.sequences[i_sequence].update(
                    self.last_element_surplus[i_elements],
                    self.element_surplus[i_elements])

            # Update the level's record of sequence activities,
            # including decay dynamics.
            self.sequence_activities[i_sequence] = np.maximum(
                    self.sequence_activities[i_sequence], new_activity) 

            # Subtract the sequence activities from the surplus 
            # that contributed to it.
            self.element_surplus[i_elements] -= (
                    self.sequence_activities[i_sequence])
            self.element_surplus[i_elements] = np.maximum(
                    self.element_surplus[i_elements], 0.)

        return self.sequence_activities

    def visualize(self):
        """
        Show the current state of the ``Level``.
        """
        print self.name
        print("element activities")
        for i_element, activity in enumerate(self.last_element_activities):
            print(" ".join(["element", str(i_element), ":", 
                            "activity ", str(activity), ",",
                            "surplus ", 
                            str(self.last_element_surplus[i_element]), ",",
                            "last surplus ", 
                            str(self.last_last_element_surplus[i_element])
                            ]))

        for i_sequence in range(self.ziptie.num_bundles):
            i_elements = self.ziptie.get_elements(i_sequence)
            print(" ".join(["    sequence", str(i_sequence), ":", 
                            str(self.last_sequence_activities[i_sequence]), 
                            ":", str(i_elements)]))
            self.sequences[i_sequence].visualize()
    '''
