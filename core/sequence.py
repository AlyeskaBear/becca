"""
The Sequence class.
"""

import numpy as np

class Sequence(object):
    """
    A set of elements and the transitions that connect them.
    
    The name "sequence" is an oversimplification. A ``Sequence`` is 
    actually a directed graph: a set of nodes and edges between
    them. The nodes are elements, and the edges are the transitions
    between them.


    Attributes
    ----------
    decay_rate : float
        See ``Level.decay_rate``.
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

        self.decay_rate = decay_rate
        self.element_indices = []
        self.num_elements = len(self.element_indices)
        self.observations = np.zeros((self.num_elements, self.num_elements))
        self.opportunities = np.zeros((self.num_elements, self.num_elements))
        self.transition_strengths = np.zeros((self.num_elements, 
                                              self.num_elements))
