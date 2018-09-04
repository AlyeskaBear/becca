"""
The StrCatTreeNode class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator
import numpy as np

import becca.str_cat_utils as scu


class StrCatTreeNode(object):
    """
    A node in the category tree for string categories.
    """
    def __init__(
        self,
        catch_all=True,
        depth=0.,
        i_input=0,
        in_crowd=None,
        n_candidates=10,
        parent=None,
        position=0.,
    ):
        """
        Create a new node.

        @param catch_all: boolean
            The root and every lower node is a catch-all. It considers
            every node a match. Only upper nodes are selective.
        @param depth: int
            The depth of this node in the tree (although height would be
            a more apt metaphor). The root is always at depth 0.
        @param i_input: int
            The index of this feature in the input vector.
        @param in_crowd: list of strings
            These are the names that belong to this node.
            If it is a catch_all, then the in_crowd will be empty.
        @param n_candidates: int
            The number of candidates to evaluate for each split.
        @param parent: StrCatTreeNode
            This is the node just higher in the tree.
        @param position: float
            The position of this node on the number line used for
            sorting them in visualization.
        """
        # lo_child, hi_child : NumCatTreeNodes
        #     The two children that belong (or will belong) to this node.
        self.lo_child = None
        self.hi_child = None
        self.parent = parent
        # leaf: boolean
        #     Is this node a leaf int the tree? All nodes are when they
        #     are first created.
        self.leaf = True
        # observations: dict of {string: int}
        #     The strings observed in this node during recent history.
        #     Each string has an associated count.
        self.observations = {}
        # n_observations: int
        #     The total number of observations for which this node was
        #     the matching leaf.
        self.n_observations = 0

        self.catch_all = catch_all
        self.depth = depth
        self.in_crowd = in_crowd
        self.i_input = i_input
        self.n_candidates = n_candidates
        self.position = position

    def __str__(self):
        """
        Create a useful string representation.

        This method is called when
        print(NumCatTreeNode) is run.

        @return: string
        """
        n_names = 3
        top_names, _ = self.top_n_names(n_names)
        node_str = (str(len(list(self.observations.keys()))) +
                    ' member categories, including  ')
        for name in top_names:
            node_str += ''.join(['\'', name, '\', '])
        node_str += '\n'
        return node_str

    def variance(self):
        """
        Calculate a variance-like measure for the set of strings observed.

        @return float
            The variance of observations so far.
        """
        return scu.variance(self.observations)

    def top_n_names(self, n_names):
        """
        Get the most commonly occurring names.

        @param n_names: int
            The number of names to return.

        @return: tuple (list of strings, list of ints)
            The top n most common names
            and the number of times each has occcurred.
        """
        names = []
        counts = []
        if len(list(self.observations.keys())) == 0:
            return names, counts
        if len(list(self.observations.keys())) <= n_names:
            n_names = len(self.observations)
        sorted_names = sorted(self.observations.items(),
                              key=operator.itemgetter(1))
        for i in range(n_names):
            names.append(sorted_names[i][0])
            counts.append(sorted_names[i][1])
        return names, counts

    def has(self, name):
        """
        Determine whether a name belongs to this node.

        @param name: string
            The string to test.

        @return: boolean
            Is name in this node?
        """
        if self.catch_all:
            return True
        elif name in self.in_crowd:
            return True
        # else
        return False

    def add(self, new_name, count=1):
        """
        Grow a leaf's collection of observed names.

        @param new_name: string
        """
        if new_name in self.observations:
            self.observations[new_name] += count
        else:
            self.observations[new_name] = count
        self.n_observations += count

    def split(self, split_names, i_input):
        """
        Create two new child nodes.

        i_input: int
            The new nodes will be associated with
            this index in the agent's input array, and the next.
        split_names: list of strings
            Where to subdivide the node to create children.
            Names in split_names the list belong to the hi_child,
            all others belong to the lo_child.
        """
        delta = 2. ** (-1. * float(self.depth + 3))

        self.hi_child = StrCatTreeNode(
            catch_all=False,
            depth=self.depth + 1,
            i_input=i_input,
            in_crowd=split_names,
            position=self.position - delta,
        )
        self.lo_child = StrCatTreeNode(
            depth=self.depth + 1,
            i_input=i_input + 1,
            position=self.position + delta,
        )

        for name, count in self.observations.items():
            if name in self.hi_child.in_crowd:
                self.hi_child.add(name, count)
            else:
                self.lo_child.add(name, count)
        self.observations = []
        self.leaf = False

    def evaluate(self, split_candidate):
        """
        For the proposed split, determine how good it will be.

        @param split_candidate: list of strings
        @return float
            The quality of the proposed split.
        """
        in_names = {}
        out_names = {}
        for name, count in self.observations.items():
            if name in split_candidate:
                in_names[name] = count
            else:
                out_names[name] = count
        return in_names.variance() + out_names.variance()

    def find_best_split(self):
        """
        Try several options and find the best split candidate.

        For now, just consider splitting on one name at a time.

        @returns a tuple of (list of strings, float)
            The in group names to split on and
            the change in variance that such a split would give.
        """
        biggest_change = 0.
        best_candidate = []
        if len(self.observations.keys()) > 1:
            original_variance = self.variance()

            # Generate candidates.
            if len(self.observations.keys()) < self.n_candidates:
                candidates = self.observations.keys()
            else:
                candidates = np.random.choice(
                    self.observations.keys(),
                    size=self.n_candidates,
                    replace=False)

            for candidate in candidates:
                new_variance = self.evaluate(candidate)
                new_change = original_variance - new_variance
                if new_change > biggest_change:
                    biggest_change = new_change
                    best_candidate = candidate

        return (best_candidate, biggest_change)
