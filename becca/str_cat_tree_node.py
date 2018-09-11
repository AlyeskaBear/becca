import operator
import numpy as np

import becca.str_cat_utils as scu


class StrCatTreeNode(object):
    """
    A node in the category tree for string categories.
    """
    def __init__(
        self,
        depth=0.,
        i_input=0,
        in_crowd=None,
        n_candidates=10,
        parent=None,
        position=0.,
    ):
        """
        Create a new node.

        When a node branches, it splits into two children,
        a high node and a low node.
        The root and every high node are catch-all nodes.
        They consider every node a match.
        Only low nodes are selective. Observations that match
        their in_crowd list belong to them.

        depth: int
            The depth of this node in the tree (although height would be
            a more apt metaphor). The root is always at depth 0.
        i_input: int
            The index of this feature in the input vector.
        in_crowd: list of strings
            These are the names that belong to this node.
            If it is a hi_child, then the in_crowd will be empty.
        n_candidates: int
            The number of candidates to evaluate for each split.
        parent: StrCatTreeNode
            This is the node just higher in the tree.
        position: float
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

        # self.catch_all = catch_all
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

        Returns
        -------
        node_str: str
        """
        n_names = 3
        top_names, _ = self.top_n_names(n_names)
        node_str = ""
        if self.leaf:
            node_str += "Leaf, "
        else:
            node_str += "branching (on "
            node_str += str(self.lo_child.in_crowd) + "), "
        if len(list(self.observations.keys())) > 0:
            node_str += "containing "
            node_str += str(len(list(self.observations.keys())))
            node_str += ' member categories, including  '
            for name in top_names:
                node_str += ''.join(['\'', name, '\', '])
        node_str += '\n'
        return node_str

    def variance(self, observations=None):
        """
        Calculate a variance-like measure for the set of strings observed.

        Returns
        -------
        float
            The variance of observations so far.
        """
        if observations is None:
            observations = self.observations
        return scu.variance(observations)

    def top_n_names(self, n_names):
        """
        Get the most commonly occurring names.

        Parameters
        ----------
        n_names: int
            The number of names to return.

        Returns
        -------
        tuple (list of strings, list of ints)
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

        When checking which branch an observation belongs to,
        the lo_child always has to be checked first.
        The hi_child thinks everything belongs to it.

        Parameters
        ----------
        name: string
            The string to test.

        Returns
        -------
        bool
            Is name in this node?
        """
        # if this is a hi_child, a catch-all node,
        # then everything matches
        if self.in_crowd is None:
            return True
        elif name in self.in_crowd:
            return True
        # else
        return False

    def add(self, new_name, count=1):
        """
        Grow a leaf's collection of observed names.

        Parameters
        ----------
        new_name: string
        """
        if new_name in self.observations:
            self.observations[new_name] += count
        else:
            self.observations[new_name] = count
        self.n_observations += count

    def split(self, split_names, i_input):
        """
        Create two new child nodes.

        Parameters
        ----------
        i_input: int
            The new nodes will be associated with
            this index in the agent's input array, and the next.
        split_names: list of strings
            Where to subdivide the node to create children.
            Names in split_names the list belong to the hi_child,
            all others belong to the lo_child.
        """
        delta = 2. ** (-1. * float(self.depth + 3))

        self.lo_child = StrCatTreeNode(
            depth=self.depth + 1,
            i_input=i_input,
            in_crowd=split_names,
            position=self.position - delta,
        )
        self.hi_child = StrCatTreeNode(
            depth=self.depth + 1,
            i_input=i_input + 1,
            position=self.position + delta,
        )

        for name, count in self.observations.items():
            if name in self.lo_child.in_crowd:
                self.lo_child.add(name, count)
            else:
                self.hi_child.add(name, count)
        self.observations = {}
        self.leaf = False
        return

    def evaluate(self, split_candidate):
        """
        For the proposed split, determine how good it will be.

        Parameters
        ----------
        split_candidate: list of strings

        Returns
        -------
        float
            The quality of the proposed split.
        """
        in_names = {}
        out_names = {}
        for name, count in self.observations.items():
            if name in split_candidate:
                in_names[name] = count
            else:
                out_names[name] = count
        return self.variance(in_names) + self.variance(out_names)

    def find_best_split(self):
        """
        Try several options and find the best split candidate.

        For now, just consider splitting on one name at a time.

        Returns
        -------
        a tuple of (list of strings, float)
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
                    best_candidate = [candidate]

        return (best_candidate, biggest_change)
