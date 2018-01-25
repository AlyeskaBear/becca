from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
 
import numpy as np


class Postprocessor(object):
    """
    The Postprocessor creates a set of discrete actions from the raw_actions
    expected by the world. All the actions it provides will be floats
    between zero and one.
    """
    def __init__(
        self,
        n_actions_per_raw=2,
        n_raw_actions=None,
    ):
        """
        Parameters
        ----------
        n_actions_per_raw: int
            The number of discretized actions per raw action. This determines
            the resolution of discretization
        n_raw_actions: int
            The number of actions that the world is expecting.
        """
        # TODO: Make discretization adaptive in number and magnitude.
        # Check for valid arguments.
        if not n_raw_actions:
            print('You have to give a number for n_raw_actions.')
            return

        self.n_raw_actions = n_raw_actions
        self.n_actions  = self.n_raw_actions * n_actions_per_raw
        
        # The mapping helps to convert from discretized actions to
        # raw actions. Each row represents a raw action.
        self.mapping = (np.cumsum(np.ones(
            (self.n_raw_actions, n_actions_per_raw)), axis=1) /
            n_actions_per_raw)

    def convert_to_raw(self, discretized_actions):
        """
        Construct a set of raw actions from the discretized_actions.

        Parameters
        ----------
        discretized_actions: array of floats
            The discrete action commands to put into effect.

        Returns
        -------
        consolidated_actions: array of floats
            The minimal set of discretized actions that were actually
            implemented.
        raw_actions: array of floats
            A set of actions for the world, each between 0 and 1.
        """
        # discretized_action commands can be between 0 and 1. This value
        # represents a proabaility that the action will be taken.
        # First, roll the dice and see which actions are commanded.
        commands = np.zeros(self.n_actions)
        commands[np.where(np.random.random_sample() <
                                   discretized_actions)] = 1

        # Find the magnitudes of each of the commanded actions.
        all_actions = self.mapping * np.reshape(
            commands, (self.n_raw_actions, -1))
        # Only keep the largest command for each action
        raw_actions = np.max(all_actions, axis=1)

        # Find the discretized representation of the raw_actions
        # that were finally issued.
        consolidated_actions = np.zeros(discretized_actions.size)
        for i_action in self.n_actions:
            if raw_actions[i_action] > 0:
                consolidated_actions[np.where(
                    all_actions[i_action, :] > 0)[0][-1]] = 1

        return consolidated_actions, raw_actions
