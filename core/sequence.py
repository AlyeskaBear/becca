"""
The Sequence class.
"""

import numpy as np

import tools

class Sequence(object):
    """
    A set of elements and the transitions that connect them.
    
    The name "sequence" is an oversimplification. A ``Sequence`` is 
    actually a directed graph: a set of nodes and edges between
    them. The nodes are elements, and the edges are the transitions
    between them.


    Attributes
    ----------
    activity : float
        The current activity of this ``Sequence``.
    decay_rate : float
        See ``Level.decay_rate``.
    element_activities : array of floats
        The activity values of each of this ``Sequence``'s elements.
    element_indices : array of ints
        An array of indices of the elements that are included 
        in this ``Sequence``.
    num_elements : int
        The total number of elements that contribute to this ``Sequence``.
    observations, opportunities, transition_strength : 2D array of floats
        These calculate and store the  strength of the transition 
        between each element in this ``Sequence``. 
        ``opportunities`` track how many times the transition has
        had the opportunity to occur.
        ``observations`` track how many times the transition has
        actually occurred.
        ``transition_strengths`` are an estimate of the probability 
        that a transition will occur, given the opportunity.

    """

    def __init__(self, decay_rate):
        """
        Set up the ``Sequence``.

        Parameters
        ----------
        decay_rate : float
            See ``Level.decay_rate``.
        """
        self.activity = 0.
        self.decay_rate = decay_rate
        self.element_indices = []
        self.num_elements = len(self.element_indices)
        self.element_activities = np.zeros(self.num_elements)
        self.observations = np.zeros((self.num_elements, self.num_elements))
        self.opportunities = np.zeros((self.num_elements, self.num_elements))
        self.transition_strengths = np.zeros((self.num_elements, 
                                              self.num_elements))

    def grow_sequence(self, new_size):
        self.num_elements = new_size
        self.element_activities = tools.pad(self.element_activities, 
                                            self.num_elements)
        self.observations = tools.pad(self.observations,
                                      (self.num_elements, self.num_elements))
        self.opportunities = tools.pad(self.opportunities, 
                                       (self.num_elements, self.num_elements))
        self.transition_strengths = tools.pad(self.transition_strengths,
                                              (self.num_elements, 
                                               self.num_elements))

    def update(self, element_activities):
        """
        Update the activities for each of the elements and the sequence.

        Parameters
        ----------
        element_activities : array of floats
            The activities of the elements that contribute to the sequence,
            in order.

        Returns
        -------
        activity : float
            The current activity of the sequence.
        """
        # Increase the number of elements in the sequence if necessary.
        if element_activities.size > self.num_elements:
            self.grow_sequence(element_activities.size)

        n = element_activities.size
        self.element_activities[:n] = element_activities[:n]
        #for i in range(element_activities.size):
        #    self.element_activities[i] = element_activities[i]

        # For now calculate as in BECCA 7, without time dynamics.
        self.activity = np.min(self.element_activities)
        return self.activity

    def learn(self):
        """
        Update the sequence's transitions.
        """
        pass


    def visualize(self):
        """
        Show the transitions within the sequence.
        """
        pass
