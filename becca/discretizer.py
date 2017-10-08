""" The Discretizer class """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from becca.cat_tree import CatTree


class Discretizer(object):
    """
    Incrementally break a set of values and/or strings into categories.
    """

    def __init__(
        self,
        base_position=0.,
        input_pool=None,
        name='discretizer',
        output_dir='output',
        split_frequency=int(1e3),
    ):
        """
        @param base_position : float
            A value used to sort discretized values when visualizing.
        @param input_pool: set of ints
            The set of indices available for assigning to new categories.
        @param name: string
            A name that helps to identify this instance of Discretizer.
        # @param n_inputs: int
        #     The number of inputs already assigned.
        @param output_dir: string
            The path (relative or absolute) in which to store any outputs.
        @param split_frequency: int
            How often should the discretizer check for category splits?
            This takes a little longer and shoudn't be done every
            time step.
        """
        self.numeric_cats = CatTree(
            base_position=base_position + .25,
            # i_input=n_inputs,
            input_pool=input_pool,
            type='numeric',
        )
        self.string_cats = CatTree(
            base_position=base_position - .25,
            # i_input=n_inputs + 1,
            input_pool=input_pool,
            type='string',
        )
        self.position = base_position

        self.name = name
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        # timestep: int
        #     The age of the discretizer in time steps. One data value
        #     is ingested per time step.
        self.timestep = 0
        self.split_frequency = split_frequency

    def __str__(self):
        """
        Represent the Discretizer as a string.
        """
        return self.string_cats.__str__() + self.numeric_cats.__str__()

    def step(
        self,
        raw_val,
        input_activities,
        input_pool,
        # n_inputs,
        # n_max_inputs,
        new_input_indices,
    ):
        """
        Incrementally build the set of categories.

        @param val: float or string or convertable to string
            The new piece of data to add to the history of observations.
        @param input_activities: array of floats
            The under-construction array of input activities
            for this time step.
        @param n_inputs : int
            The number of inputs currently assigned.
        @param n_max_inputs : int
            The maximum number of inputs possible.
        @param new_input_indices: list of tuples of (int, int)
           Tuples of (child_index, parent_index). Each time a new child
           node is added, it is recorded on this list.
        """
        self.timestep += 1

        # Determine whether the observation is string or numerical.
        is_string = True
        try:
            float_val = float(raw_val)
            if np.isnan(float_val):
                val = "NaN"
            elif np.isposinf(float_val):
                val = "positive_infinity"
            elif np.isneginf(float_val):
                val = "negative_infinity"
            else:
                val = float_val
                is_string = False
        except ValueError:
            val = str(raw_val)

        if is_string:
            self.string_cats.add(val)
            self.string_cats.categorize(val, input_activities)
        else:
            self.numeric_cats.add(val)
            self.numeric_cats.categorize(val, input_activities)

        if self.timestep % self.split_frequency == 0:
            # Try to grow new categories.
            success, n_inputs, new_input_indices = self.numeric_cats.grow(
                input_pool, new_input_indices)
            success, n_inputs, new_input_indices = self.string_cats.grow(
                input_pool, new_input_indices)

        return new_input_indices

    def find_cats(self, vals):
        """
        Find a set of categories for vals.

        @param vals: list of values to break into categories.
            These may be strings, floats or any object that can be
            converted into a float or string.
        """

        if len(vals) < 2:
            print('You need to provide more than one value.')
            print('Try again.')
            sys.exit()

        for val in vals:
            self.step(val)

    def generate(self, i_cat):
        """
        Generate a representative value of a category.
        """
        # TODO
        return i_cat * self.n_max_cats

    def report(self):
        """
        Show the current state of the discretizer.
        """
        plt.figure()
        numeric_ax = plt.subplot(2, 1, 1)
        self.numeric_cats.report(numeric_ax)
        self.string_cats.report()

        filename = os.path.join(
            self.output_dir,
            '_'.join([self.name, 'categories.png']))
        hi_res = 1200
        plt.savefig(filename, dps=hi_res)
