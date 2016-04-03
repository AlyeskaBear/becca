"""
The Level class.
"""

from numba import jit 
import numpy as np

from sequence import Sequence
from ziptie import ZipTie

class Level(object):
    """
    One level in the hierarchy of sequences.

    Attributes
    ----------
    activity_threshold : float
        Threshold below which input activity is teated as zero.
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
    last_element_activities : array of floats
        The activity levels of each of the level's elements from the 
        previous time step.
    last_sequence_activities : array of floats
        The activity levels of each of the level's sequences.
    level_index : int
        The index of this instance of ``Level``.
        0 is the first and bottom ``Level``, and it counts up from there. 
    name : str
        The name of this ``Level``.
    num_elements : int
        The maximum number of elements that this level can accept.
    num_sequences : int
        The maximum number of sequences that this level can create.
    reward_learning_rate : float
        The rate at which the reward is incrementally stepped toward 
        the most recent observed value.
        0 <= reward_learning_rate <= 1.
        Smaller rates have a longer learning time, but maintain more
        statistically accurate estimates.
    sequence_activities : array of floats
        The activity levels of each of the level's sequences.
    sequence_rewards : array of floats
        The reward associated with each of the level's sequences.
    sequences : list of Sequence
        The set of sequences created and maintained in this ``Level``.
        See ``sequence.py`` for a detailed description.
    ziptie : ZipTie
        See the pydoc for the ``ZipTie`` class in the ``ziptie.py`` module.
    """

    def __init__(self, level_index, num_elements, num_sequences):
        """
        Configure the ``Level``.

        Parameters
        ---------
        level_index : int
            See ``Level.level_index``.
        num_elements : int
            See ``Level.num_elements``.
        num_sequences : int
            See ``Level.num_sequences``.
        """
        self.num_elements = num_elements
        self.num_sequences = num_sequences
        self.level_index = level_index
        self.name = '_'.join(['level', str(self.level_index)])

        # Normalization constants
        self.input_max = np.zeros(self.num_elements)
        self.input_max_grow_time = 1e2
        self.input_max_decay_time = self.input_max_grow_time * 1e2

        self.activity_decay_rate = 1. - 1. / (2. ** self.level_index)
        self.reward_learning_rate  = 1e-3
        self.activity_threshold = .1
        
        ziptie_name = '_'.join([self.name, 'ziptie'])
        self.ziptie = ZipTie(self.num_elements, self.num_sequences,
                             name=ziptie_name, speed=self.level_index)

        self.sequences = []
        for _ in range(num_sequences):
            self.sequences.append(Sequence(self.activity_decay_rate))

        self.element_activities = np.zeros(self.num_elements)
        self.last_element_activities = np.zeros(self.num_elements)
        self.sequence_activities = np.zeros(self.num_sequences)
        self.last_sequence_activities = np.zeros(self.num_sequences)
        self.sequence_rewards = np.zeros(self.num_sequences)

    #@jit(nopython=True)
    def update_elements(self, inputs, start_index=0):
        """
        Normalize and update elements, then calculate sequence activities.

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

        Parameters
        ----------
        inputs : array of floats
            The current activity of the inputs. 
        start_index : int
            The first element to update.

        Returns
        -------
        array of floats
            The activity values of each of the sequences in the level.
        """
        epsilon = 1e-8

        if start_index + inputs.size > self.num_elements:
            print("level.Level.update_elements:")
            print("    Attempting to update out of range element activities.")

        stop_index = np.minimum(start_index + inputs.size, self.num_elements)
        # Input index
        ii = 0
        for i in range(start_index, stop_index):
            val = inputs[ii]

            # Decay the maximum value.
            self.input_max[i] += ( (val - self.input_max[i]) / 
                                   self.input_max_decay_time ) 

            # Grow the maximum value, when appropriate.
            if val > self.input_max[i]:
                self.input_max[i] += ( (val - self.input_max[i]) /
                                       self.input_max_grow_time )  

            # Scale the input by the maximum.
            val = val / (self.input_max[i] + epsilon)
            val = np.maximum(0., val)
            val = np.minimum(1., val)

            # Sparsify the cable activities to speed up processing.
            if val < self.activity_threshold:
                val = 0.

            self.element_activities[i] = np.maximum(
                    self.element_activities[i], val)
            ii += 1

        self.element_surplus = self.element_activities.copy()

        # Calculate and return the activities of each of the sequences.
        for i_sequence in np.arange(self.ziptie.num_bundles - 1, -1, -1):

            # Get the set of elements that contribute to the sequence.
            i_elements = self.ziptie.get_elements(i_sequence)

            # Calculate the activity of each sequence.
            new_activity = self.sequences[i_sequence].update(
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

    def learn_sequence_membership(self):
        """
        Learn which elements contribute to each sequence and their structure.
        """
        self.ziptie.learn(self.element_surplus, self.sequence_activities)

    def learn_sequence_transitions(self):
        """
        Within each sequence, update the estimates of the transitions..
        """
        for sequence in self.sequences:
            sequence.learn()


    def update_sequence_reward(self, reward):
        """
        Increment the expected reward value associated with each sequence.
        The size of the increment is larger when:
            1. the discrepancy between the previously learned and 
                observed reward values is larger and
            2. the sequence activity is greater.
        Another way to say this is: 
        If either the reward discrepancy is very small
        or the sequence activity is very small, there is no change.
        """
        sequence_index = 0
        for sequence in self.sequences[:self.ziptie.num_bundles]:
            self.sequence_rewards[sequence_index] += (
                    (reward - self.sequence_rewards[sequence_index]) * 
                    self.sequence_activities[sequence_index] * 
                    self.reward_learning_rate)
            sequence_index += 1

    def age(self):
        """
        Advance time by one step.
        """
        # Update the previous element activities.
        self.last_element_activities = self.element_activities
        # Decay the activity of the elements and sequences.
        self.element_activities = (self.last_element_activities * 
                                   self.activity_decay_rate)

        # Update the previous sequence activities.
        self.last_sequence_activities = self.sequence_activities
        # Decay the activity of the elements and sequences.
        self.sequence_activities = (self.last_sequence_activities * 
                                   self.activity_decay_rate)

    def visualize(self):
        """
        Show the current state of the ``Level``.
        """
        print self.name
        print('element activities')
        for i_element, activity in enumerate(self.last_element_activities):
            print(' '.join(['element', str(i_element), ':', str(activity)]))
        for i_sequence in range(self.ziptie.num_bundles):
            sequence = self.sequences[i_sequence]
            i_elements = self.ziptie.get_elements(i_sequence)
            print(' '.join(['    sequence', str(i_sequence), ':', 
                            str(self.last_sequence_activities[i_sequence]), 
                            ':', str(i_elements)]))
