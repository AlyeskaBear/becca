from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
 
import numpy as np


class Postprocessor(object):
    """
    The Postprocessor creates a set of discrete commands based on
    the actions
    expected by the world. At each time step it translates the current set
    of commands into actions. All the actions it provides will be floats
    between zero and one.
    """
    def __init__(
        self,
        n_comands_per_action=2,
        n_actions=None,
    ):
        """
        Parameters
        ----------
        n_comands_per_action: int
            The number of discretized actions per raw action. This determines
            the resolution of discretization
        n_actions: int
            The number of actions that the world is expecting.
        """
        # TODO: Make discretization adaptive in number and magnitude.
        # Check for valid arguments.
        if not n_actions:
            print('You have to give a number for n_actions.')
            return

        self.n_actions = n_actions
        self.n_commands  = self.n_actions * n_comands_per_action
        
        # The mapping helps to convert from discretized actions to
        # raw actions. Each row represents a raw action.
        self.mapping = (np.cumsum(np.ones(
            (self.n_actions, n_comands_per_action)), axis=1) /
            n_comands_per_action)

    def convert_to_actions(self, command_activities):
        """
        Construct a set of actions from the command_activities.

        Parameters
        ----------
        command_activities: array of floats
            The likelihood that each of the discrete commands will be
            put into effect.

        Returns
        -------
        consolidated_commands: array of floats
            The minimal set of commands that were actually
            implemented. Larger commands for a given action eclipse
            smaller ones.
        actions: array of floats
            A set of actions for the world, each between 0 and 1.
        """
        # command_activities can be between 0 and 1. This value
        # represents a proabaility that the action will be taken.
        # First, roll the dice and see which actions are commanded.
        commands = np.zeros(self.n_commands)
        commands[np.where(np.random.random_sample() < command_activities)] = 1

        # Find the magnitudes of each of the commanded actions.
        action_commands = self.mapping * np.reshape(
            commands, (self.n_actions, -1))
        # Only keep the largest command for each action
        actions = np.max(action_commands, axis=1)

        # Find the discretized representation of the actions
        # that were finally issued.
        consolidated_commands = np.zeros(self.n_commands)
        for i_action in range(self.n_actions):
            if actions[i_action] > 0:
                consolidated_commands[np.where(
                    action_commands[i_action, :] > 0)[0][-1]] = 1

        return consolidated_commands, actions
