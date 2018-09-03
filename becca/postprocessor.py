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
        n_commands_per_action=2,
        n_actions=None,
    ):
        """
        Parameters
        ----------
        _commands_per_action: int
            The number of discretized actions per raw action. This determines
            the resolution of discretization
        n_actions: int
            The number of actions that the world is expecting.
        """
        # TODO: Make discretization adaptive in number and magnitude.
        # TODO: Check for valid arguments.
        if not n_actions:
            print('You have to give a number for n_actions.')
            return

        self.n_actions = n_actions
        self.n_commands_per_action = n_commands_per_action
        self.n_commands = self.n_actions * self.n_commands_per_action

        # Keep a running record of recent internal values
        # for visualization.
        self.command_activities = np.zeros(self.n_commands)
        self.commands = np.zeros(self.n_commands)
        self.consolidated_commands = np.zeros(self.n_commands)
        self.previous_commands = np.zeros(self.n_commands)
        self.actions = np.zeros(self.n_actions)

        # The mapping helps to convert from discretized actions to
        # raw actions. Each row represents a raw action.
        self.mapping = (np.cumsum(np.ones(
            (self.n_actions, self.n_commands_per_action)), axis=1) /
            self.n_commands_per_action)

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
        self.consolidated_commands: array of floats
            The minimal set of commands that were actually
            implemented. Larger commands for a given action eclipse
            smaller ones.
        actions: array of floats
            A set of actions for the world, each between 0 and 1.
        """
        self.command_activities = command_activities
        # command_activities can be between 0 and 1. This value
        # represents a proabaility that the action will be taken.
        # First, roll the dice and see which actions are commanded.
        self.commands = np.zeros(self.n_commands)
        self.commands[np.where(np.random.random_sample()
                      < self.command_activities)] = 1

        # Find the magnitudes of each of the commanded actions.
        action_commands = self.mapping * np.reshape(
            self.commands, (self.n_actions, -1))
        # Only keep the largest command for each action
        self.actions = np.max(action_commands, axis=1)

        # Find the discretized representation of the actions
        # that were finally issued.
        # These are used (delayed by one time step) to let
        # the model know what it did, so that it can learn
        # the appropriate model.
        self.previous_commands = self.consolidated_commands
        self.consolidated_commands = np.zeros(self.n_commands)
        for i_action in range(self.n_actions):
            if self.actions[i_action] > 0:
                i_consolidated = (
                    i_action * self.n_commands_per_action
                    + np.where(action_commands[i_action, :] > 0)[0][-1])
                self.consolidated_commands[i_consolidated] = 1

        # TODO: Consider adding fatigue.
        # It's not clear whether it will be helpful or not.
        return self.actions
