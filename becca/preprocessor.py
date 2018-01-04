from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class InputFilter(object):
    """
    Takes a possibly large number of input candidates and selects a few
    from among them.
    """
    def __init__(
        self,
        n_inputs_final=None,
        verbose=False,
    ):
        """
        Parameters
        ----------
        n_inputs_final: int
            The number of inputs that the InputFilter is expected to return.
        verbose: boolean
        """
        # Check for valid arguments.
        if not n_inputs_final:
            print('You have to give a number for both' +
                  ' n_inputs_final.')
            return
        else:
            self.n_inputs_final = n_inputs_final

    

    def convert_to_inputs(self, actions, sensors):
        """
        Build a set of discretized inputs for the featurizer.

        Parameters
        ----------
        actions: array of floats
            The actions taken on the previous time step. These are assumed
            to be discretized already.
        sensors: list of floats, strings and/or stringifiable objects
            The sensor values from the current time step.

        Returns
        -------
        input_activities: array of floats
            The activity levels of each of the inputs, which themselves
            are discretized versions of the sensors.
        """
        raw_input_activities = np.zeros(self.input_energies.size)
        # This assumes that n_actions is constant.
        raw_input_activities[:self.n_actions] = actions
        for i_sensor, discretizer in enumerate(self.discretizers):
            raw_input_activities, self.n_inputs = discretizer.step(
                input_activities=raw_input_activities,
                n_inputs=self.n_inputs,
                raw_val=sensors[i_sensor],
            )
        input_activities = tools.fatigue(
            raw_input_activities, self.input_energies)

        # Grow input_activities and input_energies as necessary.
        if self.n_inputs > self.input_energies.size / 2:
            new_input_energies = np.ones(2 * self.n_inputs)
            new_input_energies[:self.input_energies.size] = (
                self.input_energies)
            self.input_energies = new_input_energies

        return input_activities[:self.n_inputs]
