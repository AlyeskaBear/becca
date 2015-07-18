"""
A set of functions written to take advantage of numba

After a little experimentation, it appears that numba works best
when everything is written out explicitly in loops. It even beats
numpy handily on dense matrices. It doesn't like lists or lists of lists.
It also tends to work better if you
don't use numpy functions and if you make your code as simple
as possible. 

The (nopython=True) call makes it so that if numba can't compile the code 
to C (very fast), but is forced to fall back to python instead (dead slow 
when doing loops), the function will fail and throw an error.
"""
from numba import jit 

@jit(nopython=True)
def sparsify_array1d_threshold(array1d, threshold):
    """
    Sparsify a one-dimensional array at a constant threshold
    
    In a one-dimensional array, find any values less than 
    the threshold supplied and set them equal to zero.
    """
    for row in range(array1d.size):
        if array1d[row] < threshold:
            array1d[row] = 0.

#@jit(nopython=True)
def set_dense_val(array2d, i_rows, i_cols, val):
    """
    Set values in a dense 2D array using a list of indices
    """
    for i in range(len(i_rows)):
        #print 'i', i,'i_rows[i]', i_rows[i], 'i_cols[i]', i_cols[i], 'val', val,'arr', array2d[i_rows[i], i_cols[i]] 
        array2d[i_rows[i], i_cols[i]] = val 

@jit(nopython=True)
def max_dense(array2d, results):
    """
    Find the maximum value of a dense 2D array, with its row and column

    The results array has three elements and holds
        [0] the maximum value found
        [1] the row number in which it was found
        [2] the column number in which it was found
    """
    max_val = results[0]
    i_row_max = results[1]
    i_col_max = results[2]
    for i_row in range(array2d.shape[0]): 
        for i_col in range(array2d.shape[1]): 
            if array2d[i_row, i_col] > max_val:
                max_val = array2d[i_row, i_col]
                i_row_max = i_row
                i_col_max = i_col
    results[0] = max_val
    results[1] = i_row_max
    results[2] = i_col_max

@jit(nopython=True)
def min_sparse_row_weights(i_rows, i_cols, row_weights, col_min):
    """
    Find the minimum of row weights along columns of a sparse 2D array
    
    i_rows : the row indices for the non-zero sparse 2D array elements.
    i_cols : the column indices for the non-zero sparse 2D array elements.
        All non-zero elements are assumed to be 1.
        i_rows and i_cols must be the same length.
    row_weights : a one-dimensional numpy array of weights, 
        at least as long as the number of rows in the sparse array 
    col_min : a one-dimensional numpy array of values known to be larger
        than the greatest of the row_weights, 
        at least as long as the number of columns in the sparse array 
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        if col_min[int(col)] > row_weights[int(row)]:
            col_min[int(col)] = row_weights[int(row)]

@jit(nopython=True)
def max_sparse_row_weights(i_rows, i_cols, row_weights, col_max):
    """
    Find the maximum of row weights along columns of a sparse 2D array

    Similar to min_sparse_row_weights
    col_max : a one-dimensional numpy array of values known to be smaller
        than the greatest of the row_weights, 
        at least as long as the number of columns in the sparse array 
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        if col_max[int(col)] < row_weights[int(row)]:
            col_max[int(col)] = row_weights[int(row)]

@jit(nopython=True)
def sum_sparse_row_weights(i_rows, i_cols, row_weights, col_sum):
    """
    Sum columns of a sparse 2D array using row weights  
    
    i_rows : the row indices for the non-zero sparse 2D array elements.
    i_cols : the column indices for the non-zero sparse 2D array elements.
        All non-zero elements are assumed to be 1.
        i_rows and i_cols must be the same length.
    row_weights : a one-dimensional numpy array of weights, 
        at least as long as the number of rows in the sparse array 
    col_sum : a one-dimensional numpy array of zeros, 
        at least as long as the number of columns in the sparse array 
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        col_sum[int(col)] += row_weights[int(row)]

@jit(nopython=True)
def sum_sparse_col_weights(i_rows, i_cols, col_weights, row_sum):
    """
    Sum along rows of a sparse 2D array using column weights  
    
    i_rows : the row indices for the non-zero sparse 2D array elements.
    i_cols : the column indices for the non-zero sparse 2D array elements.
        All non-zero elements are assumed to be 1.
        i_rows and i_cols must be the same length.
    col_weights : a one-dimensional numpy array of weights, 
        at least as long as the number of columns in the sparse array 
    row_sum : a one-dimensional numpy array of zeros, 
        at least as long as the number of rows in the sparse array 
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        row_sum[int(row)] += col_weights[int(col)]

@jit(nopython=True)
def max_sparse_col_weights(i_rows, i_cols, col_weights, row_max):
    """
    Find the maximum of column weights along rows in a sparse 2D array 
    
    i_rows : the row indices for the non-zero sparse 2D array elements.
    i_cols : the column indices for the non-zero sparse 2D array elements.
        All non-zero elements are assumed to be 1.
        i_rows and i_cols must be the same length.
    col_weights : a one-dimensional numpy array of weights, 
        at least as long as the number of cols in the sparse array 
    row_max : a one-dimensional numpy array of zeros, 
        at least as long as the number of rows in the sparse array 
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        if row_max[int(row)] < col_weights[int(col)]:
            row_max[int(row)] = col_weights[int(col)]

@jit(nopython=True)
def min_sparse_col_weights(i_rows, i_cols, col_weights, row_min):
    """
    Find the minimum of column weights along rows in a sparse 2D array 
    
    Similar to max_sparse_col_weights
    row_min : a one-dimensional numpy array of values known to be
        larger than the greatest of the row weights,
        at least as long as the number of rows in the sparse array 
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        if row_min[int(row)] > col_weights[int(col)]:
            row_min[int(row)] = col_weights[int(col)]
'''
@jit(nopython=True)
def nucleation_energy_decay(cable_activities,
                            nucleation_energy, 
                            NUCLEATION_ENERGY_RATE):
    """
    Decay the nucleation energy

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like 
        nucleation_energy -= (cable_activities *
                              cable_activities.T *
                              nucleation_energy * 
                              NUCLEATION_ENERGY_RATE)
    """
    for i_cable1 in range(len(cable_activities)):
        activity1 = cable_activities[i_cable1]
        if activity1 > 0.:
            for i_cable2 in range(len(cable_activities)):
                activity2 = cable_activities[i_cable2]
                if activity2 > 0.:
                    nucleation_energy[i_cable1, i_cable2] -= (
                            activity1 * activity2 * 
                            nucleation_energy[i_cable1, i_cable2] *
                            NUCLEATION_ENERGY_RATE) 
'''
@jit(nopython=True)
def nucleation_energy_gather(nonbundle_activities,
                             nucleation_energy):
    """
    Gather nucleation energy

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like 
        nucleation_energy += (nonbundle_activities * 
                              nonbundle_activities.T * 
                              NUCLEATION_ENERGY_RATE) 
    """
    for i_cable1 in range(len(nonbundle_activities)):
        activity1 = nonbundle_activities[i_cable1]
        if activity1 > 0.:
            for i_cable2 in range(len(nonbundle_activities)):
                activity2 = nonbundle_activities[i_cable2]
                if activity2 > 0.:
                    nucleation_energy[i_cable1, i_cable2] += (
                            activity1 * activity2) 

'''
@jit(nopython=True)
def agglomeration_energy_decay(cable_activities, n_bundles,
                               agglomeration_energy, 
                               AGGLOMERATION_ENERGY_RATE):
    """
    Decay the agglomeration energy

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like 
        agglomeration_energy -= (cable_activities.T *
                                 agglomeration_energy * 
                                 AGGLOMERATION_ENERGY_RATE)
    """
    for i_col in range(len(cable_activities)):
        activity = cable_activities[i_col]
        if activity > 0.:
            # Only decay bundles that have been created
            for i_row in range(n_bundles):
                agglomeration_energy[i_row, i_col] -= (
                        activity * 
                        agglomeration_energy[i_row, i_col] *
                        AGGLOMERATION_ENERGY_RATE) 
'''

@jit(nopython=True)
def agglomeration_energy_gather(bundle_activities, nonbundle_activities,
                                n_bundles, agglomeration_energy):
    """
    Decay the agglomeration energy

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like 
        coactivities = bundle_activities * nonbundle_activities.T
        agglomeration_energy += coactivities * AGGLOMERATION_ENERGY_RATE
    """
    for i_col in range(len(nonbundle_activities)):
        activity = nonbundle_activities[i_col]
        if activity > 0.:
            # Only decay bundles that have been created
            for i_row in range(n_bundles):
                if bundle_activities[i_row] > 0.:
                    coactivity = activity * bundle_activities[i_row]
                    agglomeration_energy[i_row, i_col] += coactivity 

