"""
The CatTree class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from becca.str_cat_tree_node import StrCatTreeNode
from becca.num_cat_tree_node import NumCatTreeNode


class CatTree(object):
    """
    A base class for category trees, meant to be extended for specific types.
    """
    def __init__(
        self,
        base_position=0.,
        # i_input=0,
        input_pool=None,
        split_period=int(1e2),
        split_size=1e-2,
        type='default',
        verbose=False,
    ):
        """
        @param base_position: float
            The location on a number line, used for ordering categories
            from trees into a visually interpretable tree.
        # @param i_input: int
        #     The next unassigned index in the input array.
        @param input_pool: set of int
            The input indices not currently assigned to bundles or sensors.
        @param split_period: int
            The number of time steps between attempts to create new
            categories. This is included because evaluating categories
            for splits is expensive. Performing it every time step
            would slow Becca down quite a bit.
        @param split_size: float
            A constant that determines how much of a benefit needs to
            exist to justify a category split
        @param type: string
            Which type of tree to create, 'string' or 'numeric'.
            If the argument isn't interpretable, it will default to
            numeric. Other types can be created too, if the appropriate
            <type>CatTreeNode class is created.
        @param verbose: boolean
            Over-communicate about cat_tree's internal state and workings.
        """
        # root: <type>CatTreeNode
        #     The root node of the tree. All other nodes in this tree
        #     are descended from it.
        # observation_set: <type>CatTreeNode
        #     A one-node tree that keeps a record of the set of values
        #     observed recently.
        self.verbose = verbose
        if self.verbose:
            print('\'' + type + '\'' + ' tree requested.')
        if type.lower() in ['string', 'str']:
            self.root = StrCatTreeNode(
                i_input=input_pool.pop(),
                position=base_position)
            self.observation_set = StrCatTreeNode()
            if self.verbose:
                print('string tree created.')
        else:
            self.root = NumCatTreeNode(
                i_input=input_pool.pop(),
                position=base_position)
            self.observation_set = NumCatTreeNode()
            if self.verbose:
                print('numeric tree created.')

        self.split_period = split_period
        self.split_size = split_size

        # depth: int
        #     How many nodes along this tree's longest branch?
        #     (minus the root)
        self.depth = 0
        # n_cats: int
        #     The number of categories represented in this tree.
        self.n_cats = 1

    def __str__(self):
        """
        Create a useful string representation

        This method is called when
        print(CatTree) is run.
        """
        tree_string = 'Tree nodes\n'
        for node in self.get_list():
            tree_string += '    '
            tree_string += str(node)
        return tree_string

    def count(self):
        """
        How many values have been assigned to this tree?
        """
        return self.observation_set.n_observations

    def get_list(self, leaves_only=False):
        """
        Return the nodes as a list.

        @param leaves_only: bool
            Return only the nodes of the tree that are
            eligibile for splitting.

        @return : list of nodes.
            The set of nodes in the tree.
        """
        def get_list_descend(node):
            """
            Recusively walk the tree and build a list.
            """
            if node.leaf:
                node_list.append(node)
                return

            get_list_descend(node.lo_child)
            get_list_descend(node.hi_child)
            if not leaves_only:
                node_list.append(node)
            return

        node_list = []
        get_list_descend(self.root)
        return node_list

    def get_leaf(self, value):
        """
        Retrieve the leaf associated with value.

        @param value: type determined by CatTreeNode
            The value to find in one of the tree nodes.
            It can be a number, a string, or any other type,
            depending on how CatTreeNode is implemented.

        @return CatTreeNode
            The leaf node containing value.
        """
        def get_leaf_descend(node):
            """
            Recursively descend through the tree to find the leaf node.
            """
            if node.leaf:
                return node
            elif node.lo_child.has(value):
                return get_leaf_descend(node.lo_child)
            else:
                return get_leaf_descend(node.hi_child)

        return get_leaf_descend(self.root)

    def get_lineage(self, value):
        """
        Retrieve the leaf associated with value and all its parents.

        @param value: type determined by CatTreeNode
            The value to find in one of the tree nodes.
            It can be a number, a string, or any other type,
            depending on how CatTreeNode is implemented.

        @return list of CatTreeNode
            A list of nodes, starting with the root, ending with
            the leaf node containing the value and including every
            node in between, in order.
        """
        def get_lineage_descend(node, lineage):
            """
            Recursively descend through the tree to find the lineage.
            """
            lineage.append(node)
            if node.leaf:
                return lineage
            elif node.lo_child.has(value):
                return get_lineage_descend(node.lo_child, lineage)
            else:
                return get_lineage_descend(node.hi_child, lineage)

        return get_lineage_descend(self.root, [])

    def get_parent_indices(self, node, parent_indices):
        """
        Collect the input indices of parents and grandparents, to the root.

        This is a recurrent function that walks its way up the tree, building
        out a list of ancestors' input indices.

        @param node: CatTreeNode
            The current location on the tree.
        @param parent_indices: list of ints
            The collection so far of parents' input indices.

        @return: list of ints 
            The completed list.
        """
        if node is not None:
            parent_indices.append(node.i_input)
            self.get_parent_indices(node.parent, parent_indices)

    def categorize(self, value, input_activities, discount_rate=100.):
        """
        For a value, get the category or categories it belongs to.

        @param value: type determined by CatTreeNode
            Categorize this.
        @param input_activities: array of floats
            The under-construction array of input activities
            for this time step.
        @param discount_rate: float
            A constant controlling the rate at which parent node
            activities are reduced. This allows new child nodes to gradually
            take over their parent's job, and then fade the parent activity
            out so that it doesn't interfere. A discount_rate of 100
            means that after 100 observations in the child node, the parent's
            activity will be reduced to 1/2.

        @return: array of float
            Category membership. Each element represents a category.
            Membership varies from 0. (non-member)
            to 1. (full member).
        """
        lineage = self.get_lineage(value)
        cumulative_discount = 1.
        for node in lineage[::-1]:
            input_activities[node.i_input] = cumulative_discount
            # generational_discount = 1. / (
            #     1. + node.n_observations / discount_rate)
            generational_discount = .5
            cumulative_discount *= generational_discount

    def add(self, value):
        """
        Add a value to a collection of observations and to the tree.
        """
        self.observation_set.add(value)
        self.get_leaf(value).add(value)

    def grow(self, input_pool, new_input_indices):
        # def grow(self, input_pool, n_inpu ts, n_max_inputs, new_input_indices):
        """
        Find a leaf to split.

        @param input_pool: set of ints
            Available indices to assign to categories.
        # @param n_inputs : int
        #     The total number of inputs that the discretizer is passing
        #     to the featurizer.
        # @param n_max_inputs : int
        #     The total number of inputs allowed.
        @param new_input_indices: list of tuples of (int, list of int)
           Tuples of (child_index, parent_indices). Each time a new child
           node is added, it is recorded on this list, together with
           the input indices of all its parents and grandparents.

        @return success: bool
            Was there a split worthy of splitting?
        """
        success = False
        if (
                # n_inputs < n_max_inputs and
            self.observation_set.n_observations % self.split_period == 0
        ):
            leaves = self.get_list(leaves_only=True)
            # Test splits on each leaf. Find the best.
            best_candidate = 0.
            best_leaf = None
            biggest_change = 0.
            for leaf in leaves:
                (candidate_split, change) = leaf.find_best_split()
                if change > biggest_change:
                    biggest_change = change
                    best_candidate = candidate_split
                    best_leaf = leaf

            # Check whether the best split is good enough.
            # Calculate the reduction threshold that is interesting.
            good_enough = self.observation_set.variance() * self.split_size
            if biggest_change > good_enough:
                best_leaf.split(best_candidate, input_pool)
                # TODO: pick up here
                best_leaf.lo_child.parent = best_leaf
                best_leaf.hi_child.parent = best_leaf
                self.depth = np.maximum(self.depth, best_leaf.hi_child.depth)
                
                parent_indices = []
                self.get_parent_indices(best_leaf, parent_indices)
                new_input_indices.append(
                    (best_leaf.lo_child.i_input, parent_indices))
                new_input_indices.append(
                    (best_leaf.hi_child.i_input, parent_indices))
                self.n_cats += 2
                # n_inputs += 2
                success = True
        return success, new_input_indices
