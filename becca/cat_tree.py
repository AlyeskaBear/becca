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
        i_input=0,
        split_period=1e2,
        split_size=1e-2,
        type='default',
        verbose=False,
    ):
        """
        Parameters
        ----------
        base_position: float
            The location on a number line, used for ordering categories
            from trees into a visually interpretable tree.
        i_input: int
            The next unassigned index in the input array.
        split_period: int
            The number of time steps between attempts to create new
            categories. This is included because evaluating categories
            for splits is expensive. Performing it every time step
            would slow Becca down quite a bit.
        split_size: float
            A constant that determines how much of a benefit needs to
            exist to justify a category split
        type: string
            Which type of tree to create, 'string' or 'numeric'.
            If the argument isn't interpretable, it will default to
            numeric. Other types can be created too, if the appropriate
            <type>CatTreeNode class is created.
        verbose: boolean
            Over-communicate about cat_tree's internal state and workings.
        """
        self.verbose = verbose
        if self.verbose:
            print('\'' + type + '\'' + ' tree requested.')
        if type.lower() in ['string', 'str']:
            self.root = StrCatTreeNode(
                i_input=i_input,
                position=base_position)
            self.observation_set = StrCatTreeNode()
            if self.verbose:
                print('string tree created.')
        else:
            self.root = NumCatTreeNode(
                i_input=i_input,
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

        Parameters
        ----------
        leaves_only: bool
            Return only the nodes of the tree that are
            eligibile for splitting.

        Returns
        -------
        list of nodes
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

        Parameters
        ----------
        value: type determined by CatTreeNode
            The value to find in one of the tree nodes.
            It can be a number, a string, or any other type,
            depending on how CatTreeNode is implemented.

        Returns
        -------
        CatTreeNode
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

        Parameters
        ----------
        value: type determined by CatTreeNode
            The value to find in one of the tree nodes.
            It can be a number, a string, or any other type,
            depending on how CatTreeNode is implemented.

        Returns
        -------
        list of CatTreeNode
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

        Parameters
        ----------
        node: CatTreeNode
            The current location on the tree.
        parent_indices: list of ints
            The collection so far of parents' input indices.

        Returns
        -------
        list of ints
            The completed list.
        """
        if node is not None:
            parent_indices.append(node.i_input)
            self.get_parent_indices(node.parent, parent_indices)

    def categorize(
        self,
        value,
        input_activities,
        generational_discount=.5,
    ):
        """
        For a value, get the category or categories it belongs to.

        Parameters
        ----------
        value: type determined by CatTreeNode
            Categorize this.
        input_activities: array of floats
            The under-construction array of input activities
            for this time step.
        generational_discount: float
            Between 0 and 1.
            The amount by which activities are reduced for parents.
            This allows new child nodes to take over their parent's job.

        Returns
        -------
        No explicit returns, but input activities is modified to show
            Category membership for value.
            Membership varies from 0. (non-member)
            to 1. (full member).
        """
        lineage = self.get_lineage(value)
        cumulative_discount = 1.
        for node in lineage[::-1]:
            input_activities[node.i_input] = cumulative_discount
            cumulative_discount *= generational_discount

    def add(self, value):
        """
        Add a value to the collection of observations for the corresponding
        leaf and for the tree as a whole.
        """
        self.observation_set.add(value)
        self.get_leaf(value).add(value)

    def grow(self, n_inputs):
        """
        Find a leaf to split.

        Parameters
        ----------
        n_inputs : int
            The total number of inputs being passed by the preprocessor.

        Returns
        -------
        n_inputs: int
            When a split is made, this is modified.

        """
        if self.observation_set.n_observations % self.split_period == 0:
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
                best_leaf.split(best_candidate, n_inputs)
                best_leaf.lo_child.parent = best_leaf
                best_leaf.hi_child.parent = best_leaf
                self.depth = np.maximum(self.depth, best_leaf.hi_child.depth)

                parent_indices = []
                self.get_parent_indices(best_leaf, parent_indices)
                self.n_cats += 2
                n_inputs += 2
        return n_inputs
