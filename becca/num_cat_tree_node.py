"""
The NumCatTreeNode class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class NumCatTreeNode(object):
    """
    A node in the category tree for numerical categories.
    """
    def __init__(
        self,
        bounds=None,
        depth=0.,
        i_input=0,
        n_candidates=10,
        parent=None,
        position=0.,
    ):
        """
        Create a new node.

        @param bounds: tuple(float, float)
            The upper and lower bounds for this node in the format
            (lower_bound, upper_bound). This node covers all values
            that are less than the upper bound and greater than or
            equal to the lower bound.
        @param depth: int
            The depth of this node in the tree (although height would be
            a more apt metaphor). The root is always at depth 0.
        @param i_input: int
            The index of this feature in the input vector.
        @param n_candidates: int
            The number of candidates to evaluate for each split.
        @param parent: NumCatTreeNode
            This is the node just higher in the tree.
        @param position: float
            The position of this node on the number line used for
            sorting them in visualization.
        """
        # lo_child, hi_child : NumCatTreeNodes
        #     The two children that belong (or will belong) to this node.
        self.lo_child = None
        self.hi_child = None
        # parent : NumCatTreeNode
        #     The parent node to this one. For a root node, this is None.
        self.parent = parent
        # leaf: boolean
        #     Is this node a leaf int the tree? All nodes are when they
        #     are first created.
        self.leaf = True
        # observations: list of floats
        #     The set of values observed that belong to this node.
        self.observations = []
        # n_observations: int
        #     The total number of observations for which this node was
        #     the matching leaf.
        self.n_observations = 0

        if bounds is None:
            self.lo_bound = -np.inf
            self.hi_bound = np.inf
        else:
            self.lo_bound = bounds[0]
            self.hi_bound = bounds[1]
        self.depth = depth
        self.i_input = i_input
        self.n_candidates = n_candidates
        self.position = position

    def __str__(self):
        """
        Create a useful string representation.

        This method is called when
        print(NumCatTreeNode) is run.
        """
        node_str = ('lower bound: ' +
                    str(self.lo_bound) +
                    '   upper bound: ' +
                    str(self.hi_bound) +
                    '\n')
        return node_str

    def variance(self):
        """
        Calculate variance for the set of values observed.

        @return float
            The variance of observations so far.
        """
        return np.var(np.array(self.observations))

    def has(self, value):
        """
        Determine whether a name belongs to this node.

        @param value: float
            The number to test.

        @return: boolean
            Is name in this node?
        """
        if value >= self.lo_bound and value < self.hi_bound:
            return True
        # else
        return False

    def add(self, new_value):
        """
        Grow a leaf's collection of observed values.

        @param new_value: float
        """
        self.observations.append(new_value)
        self.n_observations += 1

    def split(self, split_value, i_input):
        """
        Create two new child nodes.

        Parameters
        ----------
        i_input: int
            The index of the next leaf node to create.
        split_value: float
            Where to subdivide the node to create children.
        """
        delta = 2. ** (-1. * float(self.depth + 3))
        lo_bounds = (self.lo_bound, split_value)
        hi_bounds = (split_value, self.hi_bound)

        self.lo_child = NumCatTreeNode(
            bounds=lo_bounds,
            depth=self.depth + 1,
            i_input=i_input,
            position=self.position - delta,
        )
        self.hi_child = NumCatTreeNode(
            bounds=hi_bounds,
            depth=self.depth + 1,
            i_input=i_input + 1,
            position=self.position + delta,
        )
        for value in self.observations:
            if value < split_value:
                self.lo_child.add(value)
            else:
                self.hi_child.add(value)
        self.observations = []
        self.leaf = False
        return

    def evaluate(self, split_candidate):
        """
        For the proposed split, determine how good it will be.

        @param float: split_candidate
        @return float
            The quality of the proposed split.
        """
        vals = np.array(self.observations)
        lo_vals = vals[np.where(vals < split_candidate)]
        hi_vals = vals[np.where(vals >= split_candidate)]
        if lo_vals.size > 0:
            lo_spread = np.sum((lo_vals - np.mean(lo_vals))**2)
        else:
            lo_spread = 0.
        if hi_vals.size > 0:
            hi_spread = np.sum((hi_vals - np.mean(hi_vals))**2)
        else:
            hi_spread = 0.
        return lo_spread + hi_spread

    def find_best_split(self):
        """
        Try several options and find the best split candidate.

        @returns (float, float)
            The value to split on and the split quality.
        """

        vals = np.array(self.observations)
        original_spread = np.sum((vals - np.mean(vals))**2)

        # Generate candidates.
        lo_split_bound = np.min(self.observations)
        hi_split_bound = np.max(self.observations)
        split_candidates = lo_split_bound + (
            hi_split_bound - lo_split_bound) * (
                np.random.random_sample(size=self.n_candidates))

        biggest_change = 0.
        best_candidate = lo_split_bound
        for candidate in split_candidates:
            new_spread = self.evaluate(candidate)
            new_change = original_spread - new_spread
            if new_change > biggest_change:
                biggest_change = new_change
                best_candidate = candidate
        return (best_candidate, biggest_change)
