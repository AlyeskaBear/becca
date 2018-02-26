import numpy as np

from becca.discretizer import Discretizer
import becca.tools as tools


class Preprocessor(object):
    """
    The Preprocessor takes the raw sensor signals in and creates a set of
    inputs for the Brain to learn from.
    """
    def __init__(self, n_sensors=None):
        """
        Parameters
        ----------
        n_sensors: int
            The number of sensors that the world will be providing. These
            are the only pieces of information Becca needs about the
            world to get started.
        """
        if not n_sensors:
            print('You have to give a number for n_sensors.')
            return
        else:
            self.n_sensors = n_sensors

        # n_inputs: int
        #     The total number of inputs that the preprocessor passes on.
        self.n_inputs = 0  # self.n_commands

        # input_energies: array of floats
        #     The reservoirs of energy associated with each of the inputs.
        #     Each input is subject to fatigue.
        #     It has to be quiet for a while
        #     before it can be strongly active again.
        # Initialize it to handle many more inputs than will be needed
        # at first.
        # n_init = 10 * (self.n_commands + self.n_sensors)
        n_init = 10 * (self.n_sensors)
        self.input_energies = np.ones(n_init)

        # input_activities: array of floats
        #     The most recent set of input activities, generated from
        #     discretized sensors.
        self.input_activities = np.zeros(n_init)

        # Initialize the Discretizers that will take in the
        # (possibly continuous) sensor inputs and turn each one into
        # a set of discrete values.
        self.discretizers = []
        for i in range(self.n_sensors):
            new_discretizer = Discretizer(
                base_position=float(i) + .5,
                n_inputs=self.n_inputs,
                name='sensor_' + str(i))
            self.discretizers.append(new_discretizer)
            self.n_inputs += 2

    def convert_to_inputs(self, sensors):
        """
        Build a set of discretized inputs for the featurizer.

        Parameters
        ----------
        sensors: list of floats, strings and/or stringifiable objects
            The sensor values from the current time step.

        Returns
        -------
        input_activities: list of floats
            The activity levels associated with each of the inputs
            that the preprocessor is providing to the featurizer.
        """
        raw_input_activities = np.zeros(self.input_energies.size)
        # This assumes that n_commands is constant.
        # raw_input_activities[:self.n_commands] = commands
        for i_sensor, discretizer in enumerate(self.discretizers):
            raw_input_activities, self.n_inputs = discretizer.step(
                input_activities=raw_input_activities,
                n_inputs=self.n_inputs,
                raw_val=sensors[i_sensor],
            )
        self.input_activities = tools.fatigue(
            raw_input_activities, self.input_energies)

        # Grow input_activities and input_energies as necessary.
        if self.n_inputs > self.input_energies.size / 2:
            new_input_energies = np.ones(2 * self.n_inputs)
            new_input_energies[:self.input_energies.size] = (
                self.input_energies)
            self.input_energies = new_input_energies

        return self.input_activities[:self.n_inputs]
