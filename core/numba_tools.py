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

#@jit(nopython=True)
def set_dense_val(array2d, i_rows, i_cols, val):
    """
    Set values in a dense 2D array using a list of indices.

    Parameters
    ----------
    array2d : 2D array of floats
        The array in which to set values.
    i_rows, i_cols: array of ints
        The row and column indices of each element to change.
    val : float
        The new value to assign.
    
    Returns
    -------
    Occur indirectly by modifying ``array2d``.
    """
    for i in range(len(i_rows)):
        array2d[i_rows[i], i_cols[i]] = val 

@jit(nopython=True)
def max_dense(array2d, results):
    """
    Find the maximum value of a dense 2D array, with its row and column

    Parameters
    ----------
    array2d : 2D array of floats
        The array to find the maximum value of.
    results : array of floats, size 3
        An array for holding the results of the operation.

    Returns
    -------
    Results are returned indirectly by modifying ``results``.
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

'''
@jit(nopython=True)
def min_sparse_row_weights(i_rows, i_cols, row_weights, col_min):
    """
    Find the minimum of row weights along columns of a sparse 2D array.
    
    Parameters
    ----------
    i_rows : array of ints
        The row indices for the non-zero sparse 2D array elements.
    i_cols : array of ints
        The column indices for the non-zero sparse 2D array elements.
        All non-zero elements are assumed to be 1.
        i_rows and i_cols must be the same length.
    row_weights : 1D array of floats 
        An array of weights, 
        at least as long as the number of rows in the sparse array.
    col_min : 1D array of floats
        An array of values known to be larger
        than the greatest of the ``row_weights``, 
        at least as long as the number of columns in the sparse array.

    Results
    -------
    Returned indirectly by modifying ```col_min``.
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        if col_min[int(col)] > row_weights[int(row)]:
            col_min[int(col)] = row_weights[int(row)]
'''
'''
@jit(nopython=True)
def max_sparse_row_weights(i_rows, i_cols, row_weights, col_max):
    """
    Find the maximum of row weights along columns of a sparse 2D array

    Parameters
    ----------
    Similar to ``min_sparse_row_weights``
    col_max : 1D array of floats
        An array of values known to be smaller
        than the greatest of the row_weights, 
        at least as long as the number of columns in the sparse array.

    Results
    -------
    Returned indirectly by modifying ```col_max``.
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        if col_max[int(col)] < row_weights[int(row)]:
            col_max[int(col)] = row_weights[int(row)]
'''
'''
@jit(nopython=True)
def sum_sparse_row_weights(i_rows, i_cols, row_weights, col_sum):
    """
    Sum columns of a sparse 2D array using row weights  
    
    Parameters
    ----------
    Similar to ``min_sparse_row_weights``
    col_sum : 1D array of floats
        An array of zeros, 
        at least as long as the number of columns in the sparse array 

    Results
    -------
    Returned indirectly by modifying ```col_sum``.
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        col_sum[int(col)] += row_weights[int(row)]

@jit(nopython=True)
def sum_sparse_col_weights(i_rows, i_cols, col_weights, row_sum):
    """
    Sum along rows of a sparse 2D array using column weights  
    
    Parameters
    ----------
    Similar to ``min_sparse_row_weights``
    row_sum : 1D array of floats
        An array of zeros, 
        at least as long as the number of rows in the sparse array 

    Results
    -------
    Returned indirectly by modifying ```row_sum``.
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        row_sum[int(row)] += col_weights[int(col)]
'''
'''
@jit(nopython=True)
def max_sparse_col_weights(i_rows, i_cols, col_weights, row_max):
    """
    Find the maximum of column weights along rows in a sparse 2D array 
    
    Parameters
    ----------
    Similar to ``min_sparse_row_weights``
    row_max : 1D array of floats
        An array of zeros, 
        at least as long as the number of rows in the sparse array 

    Results
    -------
    Returned indirectly by modifying ``row_max``.
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        if row_max[int(row)] < col_weights[int(col)]:
            row_max[int(row)] = col_weights[int(col)]
'''
'''
@jit(nopython=True)
def min_sparse_col_weights(i_rows, i_cols, col_weights, row_min):
    """
    Find the minimum of column weights along rows in a sparse 2D array 
    
    Parameters
    ----------
    Similar to ``min_sparse_row_weights``
    row_min : 1D array of floats
        An array of values known to be
        larger than the greatest of the row weights,
        at least as long as the number of rows in the sparse array 

    Results
    -------
    Returned indirectly by modifying ```row_min``.
    """
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        if row_min[int(row)] > col_weights[int(col)]:
            row_min[int(row)] = col_weights[int(col)]
'''
'''
# Array creation is supported in jit's object mode, but not its nopython mode.
@jit
#@jit(nopython=True)
def mean_sparse_col_weights(i_rows, i_cols, col_weights, row_mean):
    """
    Find the mean of column weights along rows in a sparse 2D array 
    
    Parameters
    ----------
    Similar to ``mean_sparse_row_weights``
    row_mean : 1D array of floats
        The means by row of the weights listed in ``i_rows`` and ``i_cols``.
        At least as long as the number of rows in the sparse array.

    Results
    -------
    Returned indirectly by modifying ```row_mean``.
    """
    total = np.zeros(len(row_mean))
    n = np.zeros(len(row_mean))
    for i in range(len(i_rows)):
        row = i_rows[i]
        col = i_cols[i]
        total[int(row)] += col_weights[int(col)]
        n[int(row)] += 1.
    for j in range(len(row_mean)):
        if n[j] > 0.:
            row_mean[j] = total[j] / n[j]
'''
@jit(nopython=True)
def find_bundle_activities(i_rows, i_cols, cables, bundles, threshold):
    """
    Use a greedy method to sparsely translate cables to bundles.

    Start at the last bundle added and work backward to the first. 
    For each, calculate the bundle activity by finding the minimum
    value of each of its constituent cables. Then subtract out
    the bundle activity from each of its cables.
     
    Parameters
    ----------
    bundles : 1D array of floats
        An array of bundle activity values. Initially it is all zeros.
    cables : 1D array of floats
        An array of cable activity values. 
    i_rows : array of ints
        The row indices for the non-zero sparse 2D array..
    i_cols : array of ints
        The column indices for the non-zero sparse 2D array elements.
        All non-zero elements are assumed to be 1.
        i_rows and i_cols must be the same length.
        Each column represents a cable and each row represents a bundle.
        The 2D array is a map from cables to bundles.
    threshold : float
        The amount of bundle activity below which, we just don't care.

    Results
    -------
    Returned indirectly by modifying ```cables``. These are the residual
    cable activities that are not represented by any bundle activities.
    """
    large = 1e10
    
    # Iterate over each bundle, that is, over each row.
    i = len(i_rows) - 1
    row = i_rows[i]
    while row > -1:

        # For each bundle, find the minimum cable activity that 
        # contribues to it.
        min_val = large
        i_bundle = i
        while i_rows[i] == row and i > -1:
            col = i_cols[i]
            val = cables[col]
            if val < min_val:
                min_val = val  
            i -= 1

        # Set the bundle activity.
        bundles[row] = min_val

        # Subtract the bundle activity from each of the cables.
        i = i_bundle
        while row == i_rows[i] and i > -1:
            col = i_cols[i]
            cables[col] -= min_val
            i -= 1
        
        row -= 1

@jit(nopython=True)
def find_sparse_bundle_activities(i_rows, i_cols, cables, bundles, 
                                  weights, threshold):
    """
    Use a greedy method to sparsely translate cables to bundles.

    Start at the last bundle added and work backward to the first. 
    For each, calculate the bundle activity by finding the minimum
    value of each of its constituent cables. Then subtract out
    the bundle activity from each of its cables.
     
    Parameters
    ----------
    bundles : 1D array of floats
        An array of bundle activity values. Initially it is all zeros.
    cables : 1D array of floats
        An array of cable activity values. 
    i_rows : array of ints
        The row indices for the non-zero sparse 2D array..
    i_cols : array of ints
        The column indices for the non-zero sparse 2D array elements.
        All non-zero elements are assumed to be 1.
        i_rows and i_cols must be the same length.
        Each column represents a cable and each row represents a bundle.
        The 2D array is a map from cables to bundles.
    threshold : float
        The amount of bundle activity below which, we just don't care.
    weights : array of floats
        A multiplier for how strongly the activity of each bundle should
        be considered when greedily selecting the next one to activate.

    Results
    -------
    Returned indirectly by modifying ```cables``. These are the residual
    cable activities that are not represented by any bundle activities.
    """
    large = 1e10
    max_vote = large

    # Repeat this process until the residual cable activities don't match
    # any bundles well.
    while max_vote > threshold:
        # Initialize the loop that greedily looks for the most strongly
        # activated bundle.
        max_vote = 0.
        best_val = 0.
        best_bundle = 0
        # This is the index in ``i_row`` and ``i_col`` where the 
        # current bundle's cable constituents are listed. Cable indices
        # are assumed to be contiguous and bundles are assumed to be
        # listed in ascending order of index.
        i_best_bundle = 0

        # Iterate over each bundle, that is, over each row.
        i = len(i_rows) - 1
        row = i_rows[i]
        while row > -1:

            # For each bundle, find the minimum cable activity that 
            # contribues to it.
            min_val = large
            n_cables = 0.
            i_bundle = i
            while i_rows[i] == row and i > -1:
                col = i_cols[i]
                val = cables[col]
                if val < min_val:
                    min_val = val  
                n_cables += 1.
                i -= 1
    
            # The strength of the vote for the bundle is the minimum cable
            # activity multiplied by the number of cables. This weights
            # bundles with many member cables more highly than bundles
            # with few cables. It is a way to encourage sparsity and to 
            # avoid creating more bundles than necessary.
            vote = min_val * (1. + .1 * (n_cables - 1.)) * (1. + weights[row])

            # Update the winning bundle if appropriate.
            if vote > max_vote:
                max_vote = vote
                best_val = min_val
                i_best_bundle = i_bundle
                best_bundle = row

            # Move on to the next bundle.
            row -= 1

        if best_val > 0.:
            # Set the bundle activity.
            bundles[best_bundle] = best_val

            # Subtract the bundle activity from each of the cables.
            # Using ``i_best_bundle`` lets us jump right to the place in 
            # the list of indices where the cables for the winning bundle
            # are listed.
            i = i_best_bundle
            while i_rows[i] == best_bundle and i > -1:
                col = i_cols[i]
                cables[col] -= best_val
                i -= 1

@jit(nopython=True)
def nucleation_energy_gather(nonbundle_activities, nucleation_energy):
    """
    Gather nucleation energy.

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like 
        nucleation_energy += (nonbundle_activities * 
                              nonbundle_activities.T * 
                              nucleation_energy_rate) 
    Parameters
    ----------
    nonbundle_activities : array of floats
        The current activity of each input feature that is not explained by
        or captured in a bundle.
    nucleation_energy : 2D array of floats
        The amount of nucleation energy accumulated between each pair of
        input features.

    Results
    -------
    Returned indirectly by modifying ```nucleation_energy``.
    """
    for i_cable1 in range(len(nonbundle_activities)):
        activity1 = nonbundle_activities[i_cable1]
        if activity1 > 0.:
            for i_cable2 in range(len(nonbundle_activities)):
                activity2 = nonbundle_activities[i_cable2]
                if activity2 > 0.:
                    nucleation_energy[i_cable1, i_cable2] += (
                            activity1 * activity2) 

@jit(nopython=True)
def agglomeration_energy_gather(bundle_activities, nonbundle_activities,
                                n_bundles, agglomeration_energy):
    """
    Accumulate the energy binding a new feature to an existing bundle..

    This formulation takes advantage of loops and the sparsity of the data.
    The original arithmetic looks like 
        coactivities = bundle_activities * nonbundle_activities.T
        agglomeration_energy += coactivities * agglomeration_energy_rate

    Parameters
    ----------
    bundle_activities : array of floats
        The activity level of each bundle.
    nonbundle_activities : array of floats
        The current activity of each input feature that is not explained by
        or captured in a bundle.
    n_bundles : int
        The number of bundles that have been created so far.
    agglomeration_energy : 2D array of floats
        The total energy that has been accumulated between each input feature
        and each bundle.

    Results
    -------
    Returned indirectly by modifying ```agglomeration_energy``.
    """
    for i_col in range(len(nonbundle_activities)):
        activity = nonbundle_activities[i_col]
        if activity > 0.:
            # Only decay bundles that have been created
            for i_row in range(n_bundles):
                if bundle_activities[i_row] > 0.:
                    coactivity = activity * bundle_activities[i_row]
                    agglomeration_energy[i_row, i_col] += coactivity 

@jit(nopython=True)
#def get_decision_values(probabilities, observations, opportunities,
def get_decision_values(observations, opportunities,
                        curiosities, features, 
                        #reward, decision_values, feature_importance):
                        reward, decision_values, feature_importance,
                        live_elements, live_features):
    """
    Estimate the value associated with each potential decision.

    Calculate the value function for each feature-decision pair, given
    the current set of rewards and the learned transition model.
    The value function (or Q function, as it is often abbreviated in
    reinforcement learning) estimates the value to be gleaned from
    each action, given a state.
    
    The Q function is calculated by finding the expected reward and
    goal reward associated with each outcome. For each
    feature-decision-outcome transition, the reward is weighted
    by the transition probability. For each feature-decision pair,
    the value is the maximum of the transition-weighted rewards.
    For each decision, the value is the maximum of the 
    feature-decision values over all features
    when each are weighted by the current feature activities. 
    
    Feature importance is used to focus attention on features that are
    most likely to result in a reward or are relevant to current goals.
    For each feature, the value is the maximum of the
    feature-decision values over all possible decisions. It is a way
    to measure how much a particular feature allows reward or goals
    to be achieved.

    Parameters
    ----------
    probabilities, curiosities : array of floats
        See the documentation for the ``Cerebellum`` class for a longer
        description.
    decision_values : 1D array of floats
        Scores associated with each of the possible decisions the 
        ganglia could make. It should be initialized to a one-dimensional
        array of all zeros, before being passed in. Its length should 
        be equal to the number of actions plus the number of features. 
    feature_importance : 1D array of floats
        Weights associated with each feature, reflecting how much 
        reward that feature can help BECCA find. It should be initialized
        as a one-dimensional array of large negative values. Its length
        should be equal to the number of features.
    features : array of floats
        The current set of feature activities.
    live_elements, live_features : array of floats
        Indicator arrays showing which elements and features have ever 
        been active. Elements and features that have never been active
        are assumed to be unused. Indexing them allows the loops to skip them 
        and run much faster.
    reward : array of floats
        The current set of reward values placed on each of the feature
        activities. This includes both learned reward and temporarily 
        assigned goal value.

    Returns
    -------
    This function returns its results by modifying ``decision_values``
    and ``feature_importance``.
    """
    small = 1e-3
    low = 1e-10
    #(I, J, K) = probabilities.shape
    (I, J, K) = observations.shape
    for j in range(J):
        if live_elements[j] == 1.:
            for i in range(I):
                # Skipping the iteration for small values takes advantage of
                # any sparseness present and speeds up computation considerably.
                if live_features[i] == 1. and features[i] > small:
                    best_Q_value = low
                    for k in range(K):
                        if live_features[k] == 1.:
                            # Find the highest value transition.
                            #weighted_observations = observations[i,j,k]
                            probability = (observations[i,j,k] / 
                                           opportunities[i,j])
                            #weighted_observations = observations[i,j,k] * (1. - 1./
                            #     (1. + observations[i,j,k] ** 2))
                            q = reward[k] * probability
                                 #probabilities[i,j,k])
                                 #weighted_observations / (opportunities[i,j] + low))
                            if q > best_Q_value:
                                best_Q_value = q

                    # Find the highest value feature-decision pair for each 
                    # decision. This also includes the curiosity associated with 
                    # that pair, and is weighted by the feature's activity.
                    action_value = ( (best_Q_value + curiosities[i,j]) * 
                                     features[i])
                    if action_value > decision_values[j]:
                        decision_values[j] = action_value

                    # Find the highest value feature-decision pair for each 
                    # feature.
                    if best_Q_value > feature_importance[i]:
                        feature_importance[i] = best_Q_value
    return 
                    
@jit(nopython=True)
#def cerebellum_learn(opportunities, observations, probabilities, curiosities,
def cerebellum_learn(opportunities, observations, curiosities,
                     training_context, training_goals, training_results, 
                     current_context, goals, curiosity_rate, satisfaction,
                     live_elements, live_features):
    """
    Use this time step's information to help the ``Cerebellum`` learn.
    
    Parameters
    ----------
    opportunities, observations, probabiities, curiosities : arrays of floats
        Refer to the documentation for the ``Cerebelum`` class.
    curiosity_rate : float
        A constant controlling the rate at which curiosity accrues.
    current_context : array of floats
        The set of feature activities.
    goals : array of floats
        The concatenation of the actions commanded this time step and
        the array of unfulfilled feature goals.
    live_elements, live_features : array of floats
        Indicator arrays showing which elements and features have ever 
        been active. Elements and features that have never been active
        are assumed to be unused. Indexing them allows the loops to skip them 
        and run much faster.
    satisfaction : float
        The filtered reward history. 
    training_context, training_goals, training_results : arrays of floats
        The set of feature activities, goals, and results from a time step 
        in the recent past. These aren't entirely current, because it
        takes a few time steps to wait around and see what all the 
        results of the goals and actions are going to be.

    Returns
    -------
    This function returns its results indirectly by modifying the contents
    of ``opportunities``, ``observations``, ``probabilities``, and
    ``curiosities``.
    """
    small = 1e-3

    # Note which curiosities have been satisified.
    #(I, J, K) = probabilities.shape
    (I, J, K) = observations.shape
    for i in range(I):
        # Skipping the iteration for small values takes advantage of
        # any sparseness present and speeds up computation considerably.
        if live_features[i] == 1. and current_context[i] > small:
            for j in range(J):
                if live_elements[j] == 1. and abs(goals[j]) > small:
                    curiosities[i,j] = curiosities[i,j] * ( 
                            1. - (current_context[i] * abs(goals[j])))

    for i in range(I):
        if live_features[i] == 1. and training_context[i] > small:
            for j in range(J):
                if live_elements[j] == 1.:
                    if abs(training_goals[j]) > small:
                        # Update opportunities, an upper bound on how many 
                        # times a given outcome could have occurred.
                        opportunities[i,j] += (training_context[i] * 
                                               training_goals[j])

                        for k in range(K):
                            if (live_features[k] == 1. and 
                                training_results[k] > small):
                                # Update observations, the actual 
                                # number of times
                                # each outcome has occurred.
                                observations[i,j,k] += ( training_context[i] * 
                                                         training_goals[j] *
                                                         training_results[k] )
                            # Use a strict frequentist interpretation 
                            # of probability: observations over opportunities.
                            #probabilities[i,j,k] = (observations[i,j,k] / 
                            #                        opportunities[i,j])

                    # Add an estimate of the uncertainty. It's conceptually 
                    # similar to standard error estimates for 
                    # the normal distribution: 1/sqrt(N), except that this
                    # estimate is far more liberal: 1/N**2.
                    # This allows BECCA to pretend it is more confident than
                    # it really is, and explore less. BECCA can get away with 
                    # this because its purpose is not to fully characterize its
                    # world, but rather the narrower goal of accumulating as much
                    # reward as possible.
                    uncertainty = 1. / (1. + opportunities[i,j])

                    # Increment the curiosities by the uncertainties, weighted
                    # by the feature activities.
                    # TODO: Weight by the how much reward has been 
                    # received recently.
                    curiosities[i,j] += (curiosity_rate * 
                                         uncertainty * 
                                         training_context[i] *
                                         (1. - curiosities[i,j]) *
                                         (1. - satisfaction) )
