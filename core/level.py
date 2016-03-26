"""
The Level class.
"""

import numpy as np

from sequence import Sequence

class Level(object):
    """
    One level in the hierarchy of sequences.

    Attributes
    ----------
    decay_rate : float
        The fraction that sequence activities decay.
        0 <= decay_rate <= 1.
    element_activities : array of floats
        The activity levels of each of the level's elements.
    last_element_activities : array of floats
        The activity levels of each of the level's elements from the 
        previous time step.
    level_index : int
        The index of this instance of ``Level``.
        0 is the first and bottom ``Level``, and it counts up from there. 
    next_sequence : int
        The index of the next sequence to be populated with elements.
        All the sequences at higher indices are empty.
    num_elements : int
        The maximum number of elements that this level can accept.
    num_sequences : int
        The maximum number of sequences that this level can create.
    sequence_activities : array of floats
        The activity levels of each of the level's sequences.
    sequences : list of Sequence
        The set of sequences created and maintained in this ``Level``.
        See ``sequence.py`` for a detailed description.
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
        self.decay_rate = 1. - 1. / (2. ** self.level_index)

        self.sequences = []
        for _ in range(num_sequences):
            self.sequences.append(Sequence(self.decay_rate))
        self.next_sequence = 0        

        self.element_activities = np.zeros(self.num_elements)
        self.last_element_activities = np.zeros(self.num_elements)
        self.sequence_activities = np.zeros(self.num_sequences)

    def update_elements(self, inputs, start_index=0):
        """
        Update the activities of the elements.
        """
        if start_index + inputs.size > self.num_elements:
            print("level.Level.update_elements:")
            print("    Attempting to update out of range element activities.")
        
        stop_index = np.minimum(start_index + inputs.size, self.num_elements)
        input_index = 0
        for i in range(start_index, stop_index):
            self.element_activities[i] = np.maximum(
                    self.element_activities[i],
                    inputs[input_index])
            input_index += 1

    def get_sequence_activities(self):
        """
        Calculate and return the activities of each of the sequences.
        """
        sequence_index = 0
        for sequence in self.sequences[:self.next_sequence]:
            new_activity = sequence.get_activity()
            self.sequence_activities[sequence_index] = maximum(
                    self.sequence_activities[sequence_index], new_activity) 
            sequence_index += 1

        return self.sequence_activities

    def age(self):
        """
        Advance time by one step.
        """
        # Update the previous element activities.
        self.last_element_activities = self.element_activities
        # Decay the activity of the elements and sequences.
        self.element_activities = (self.last_element_activities * 
                                   self.decay_rate)
