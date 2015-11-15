""" 
Constants and functions for use across the BECCA core
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Shared constants
epsilon = sys.float_info.epsilon
big = 10 ** 20
max_int16 = np.iinfo(np.int16).max

# Colors for plotting
dark_grey = (0.2, 0.2, 0.2)
light_grey = (0.9, 0.9, 0.9)
red = (0.9, 0.3, 0.3)
# BECCA pallette
copper_highlight = (253./255., 249./255., 240./255.)
light_copper = (242./255., 166./255., 108./255.)
copper = (175./255., 102./255, 53./255.)
dark_copper = (132./255., 73./255., 36./255.)
copper_shadow = (25./255., 22./255, 20./255.)
oxide = (20./255., 120./255., 150./255.)

'''
def map_one_to_inf(a):
    """ 
    Map values from [0, 1] onto [0, inf) and from [-1, 0] onto (-inf, 0].

    The mapping is done as follows:
            
                   1
            a' = ------- - 1 , for a >= 0
                 1 - |a|

                   -1
            a' = ------- + 1 , for a < 0
                 1 - |a|

    No error checking is performed. It is the user's responsibility to 
    make sure that |a| <= 1.

    Parameters
    ----------
    a : float or array of floats
        Each element in ``a`` is on [-1, 1] and is mapped individually.

    Returns
    -------
    a_prime : float or array of floats 
        The same size as ``a``.
    """
    a_prime = np.sign(a) / (1 - np.abs(a) + epsilon) - np.sign(a)
    return a_prime

def map_inf_to_one(a_prime):
    """ 
    Map values from [0, inf) onto [0, 1] and from  (-inf, 0] onto [-1, 0].

    The mapping is done as follows:
            
                       1
            a = 1 - ------- , for a' >= 0
                    |a'| + 1

                        1
            a = -1 + ------- , for a' < 0
                     |a'| + 1

    Parameters
    ----------
    a_prime : float or array of floats
        Each element in ``a_prime`` is mapped individually.

    Returns
    -------
    a : float or array of floats 
        The same size as ``a_prime``, mapped to [-1,1].
    """
    a = np.sign(a_prime) * (1 - 1 / (np.abs(a_prime) + 1))
    return a

def bounded_sum(a, axis=0):
    """ 
    Sum elements nonlinearly, such that the total is less than 1.
    
    To be more precise, as long as all elements in a are between -1
    and 1, their sum will also be between -1 and 1. i

    Parameters
    ----------
    a : list of floats or n-dimensional array of floats
        The elements to sum.
    axis : int
        The axis along which to sum the array, if ``a`` is an array.
        Default is 0.
    """ 
    if type(a) is list:
        total = map_one_to_inf(a[0])
        for item in a[1:]:
            total += map_one_to_inf(item)
        return map_inf_to_one(total)
    else:
        # handle the case where a is a one-dimensional array
        if len(a.shape) == 1:
            a = a[:, np.newaxis]
        bounded_total = map_inf_to_one(np.sum(map_one_to_inf(a), axis=axis))
        return bounded_total[:,np.newaxis]

def bounded_sum2(a, b):
    """ 
    Sum elements nonlinearly, such that the total is less than 1.
    
    This is the same as bounded_sum, except that it expects two arrays of
    equal size as inputs. The arrays are summed element-wise.
    """ 
    total = map_one_to_inf(a)
    total += map_one_to_inf(b)
    return map_inf_to_one(total)
'''

def pad(a, shape, val=0., dtype=float):
    """
    Pad a numpy array to the specified shape.
    
    Use val (default 0) to fill in the extra spaces. 

    Parameters
    ----------
    a : array of ints or floats
        The array to pad.
    shape : int, list of ints or tuple of ints
        The shape to which to pad ``a``.
        If any element of shape is 0, that size remains unchanged in 
        that axis. If any element of shape is < 0, the size of ``a`` in that
        axis is incremented by the magnitude of that value.
    val : float
        The value with which to pad ``a``. Default is 0.
    dtype : dtype
        The data type with which to pad ``a``.  
         
    Returns
    -------
    padded : array of ints or floats
        The padded version of ``a``.
    """
    # For  padding a 1D array
    if isinstance(shape, (int, long)):
        if shape <= 0:
            rows = a.size - shape
        else:
            rows = shape
            if rows < a.size:
                print ' '.join(['a.size is', str(a.size), 
                                ' but trying to pad to ', str(rows), 'rows.'])
                return a
        # Handle the case where a is a one-dimensional array
        padded = np.ones(rows, dtype=dtype) * val
        padded[:a.size] = a
        return padded

    # For padding a n-D array
    new_shape = shape
    n_dim = len(shape)
    if n_dim > 4:
        print(''.join([str(n_dim), ' dimensions? Now you\'re getting greedy']))
        return a

    for dim in range(len(shape)):
        if shape[dim] <= 0:
            new_shape[dim] = a.shape[dim] - shape[dim]
        else:
            if new_shape[dim] < a.shape[dim]:
                print(''.join(['The variable shape in dimension ', 
                               str(dim), ' is ', str(a.shape[dim]), 
                               ' but you are trying to pad to ', 
                               str(new_shape[dim]), '.']))
                print('You aren\'t allowed to make it smaller.') 
                return a

    padded = np.ones(new_shape, dtype=dtype) * val
    if len(new_shape) == 2:
        padded[:a.shape[0], :a.shape[1]] = a
        return padded
    if len(new_shape) == 3:
        padded[:a.shape[0], :a.shape[1], :a.shape[2]] = a
        return padded
    # A maximum of 4 dimensions is enforced.
    padded[:a.shape[0], :a.shape[1], :a.shape[2], :a.shape[3]] = a
    return padded

def str_to_int(exp):
    """ 
    Convert a string to an integer.

    The method is primitive, using a simple hash based on the 
    ordinal value of the characters and their position in the string.

    Parameters
    ----------
    exp : str
        The string expression to convert to an int.

    Returns
    -------
    sum : int
        An integer that is likely (though not extremely so) to be unique
        within the scope of the program.
    """ 
    sum = 0
    for i, character in enumerate(exp):
        sum += i + ord(character) + i * ord(character)
    return sum

def timestr(timestep, s_per_step=.25, precise=True): 
    """ 
    Convert the number of time steps into an age.
    
    Parameters
    ----------
    timestep : int
        The age in time steps.
    s_per_step : float
        The duration of each time step in seconds.
    precise : bool
        If True, report the age down to the second.
        If False, just report the most significant unit of time.
        Default is True

    Returns
    -------
    time_str : str
        The age in string format, including, as appropriate, years, 
        months, days, hours, minutes, and seconds.
    """
    # Start by calculating the total number of seconds.
    total_sec = timestep * s_per_step
    sec = int(np.mod(total_sec, 60.))
    time_str = ' '.join([str(sec), 'sec'])

    # If necessary, calculate the total number of minutes.
    total_min = int(total_sec / 60)
    if total_min == 0:
        return time_str
    min = int(np.mod(total_min, 60.))
    if precise:
        time_str = ' '.join([str(min), 'min', time_str])
    else:
        time_str = ' '.join([str(min), 'min'])

    # If necessary, calculate the total number of hours.
    total_hr = int(total_min / 60)
    if total_hr == 0:
        return time_str
    hr = int(np.mod(total_hr, 24.))
    if precise:
        time_str = ' '.join([str(hr), 'hr', time_str])
    else:
        time_str = ' '.join([str(hr), 'hr'])

    # If necessary, calculate the total number of days.
    total_day = int(total_hr / 24)
    if total_day == 0:
        return time_str
    day = int(np.mod(total_day, 30.))
    if precise:
        time_str = ' '.join([str(day), 'dy', time_str])
    else:
        time_str = ' '.join([str(day), 'dy'])

    # If necessary, calculate the total number of months.
    total_mon = int(total_day / 30)
    if total_mon == 0:
        return time_str
    mon = int(np.mod(total_mon, 12.))
    if precise:
        time_str = ' '.join([str(mon), 'mo', time_str])
    else:
        time_str = ' '.join([str(mon), 'mo'])

    # If necessary, calculate the total number of years.
    yr = int(total_mon / 12)
    if yr == 0:
        return time_str
    if precise: 
        time_str = ' '.join([str(yr), 'yr', time_str])
    else:
        time_str = ' '.join([str(yr), 'yr'])

    return time_str

def get_files_with_suffix(dir_name, suffixes):
    """ 
    Get all of the files with a given suffix in dir recursively.

    Parameters
    ----------
    dir_name : str
        The path to the directory to search.
    suffixes : list of str
        The set of suffixes for which files are being collected.

    Returns
    -------
    found_filenames : list of str
        The filenames, including the local path from ``dir_name``.
    """
    found_filenames = []
    for localpath, directories, filenames in os.walk(dir_name):
        for filename in filenames:
            for suffix in suffixes:
                if filename.endswith(suffix):
                    found_filenames.append(os.path.join(localpath, filename))
    found_filenames.sort()
    return found_filenames

def visualize_array(image_data, label='data_figure'):
    """ 
    Produce a visual representation of the image_data matrix.

    Parameters
    ----------
    image_data : 2D array of floats
        The pixel values to make into an image.
    label : str
        The string label to affix to the image. It is used both
        to generate a figure number and as the title. 
    """
    # Treat nan values like zeros for display purposes
    image_data = np.nan_to_num(np.copy(image_data))
     
    fig = plt.figure(str_to_int(label))
    # Diane made the brilliant suggestion to leave this plot in color. 
    # It looks much prettier.
    plt.summer()
    im = plt.imshow(image_data)
    im.set_interpolation('nearest')
    plt.title(label)
    plt.xlabel('Max = {0:.3}, Min = {1:.3}'.format(np.max(image_data), 
                                                   np.min(image_data)))
