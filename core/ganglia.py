"""
The Ganglia class
"""
import numpy as np

class Ganglia(object):
    """
    Plan goals and choose actions for the ``brain``.

    The ``Ganglia`` class is analogous to the basal ganglia in the human brain,
    a collection if fascinating structures and circuits deep in one of
    the oldest parts of the brain. No one understands exactly what they
    do but when they are damaged by trauma or disease, people lose 
    the ability to choose and execute actions well. I'm going pretty far out
    on a neuroscientific limb by suggesting that basal ganglionic functions
    include simulation, planning, selection, inhibition, and execution of
    volitional activity, but it is my best guess right now and I'm going
    to run with it.

    Attributes
    ----------
    actions : array of floats
        See docstring for ``brain.py.step.returns``.

    """

    def __init__(self, num_sensors, num_actions):
        """
        Configure the ``ganglia``.
        
        Parameters
        ----------
        num_sensors, num_actions : int
            See docstring for ``brain.py``.
        """
        self.actions = np.zeros(num_actions)
        
    def step(self, features):
        """
        Choose which , if any, actions to take on this time step

        """
        # Choose a single random action 
        random_action = True
        if random_action:
            self.actions = np.zeros(self.actions.shape)
            random_action_index = np.random.randint(self.actions.size)
            self.actions[random_action_index] = 1. 

        return self.actions
