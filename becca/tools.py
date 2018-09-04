"""
Constants and functions for use across the Becca core.
"""

from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Shared constants
epsilon = sys.float_info.epsilon
big = 10 ** 20
max_int16 = np.iinfo(np.int16).max


def pad(arr, shape, val=0., dtype=float):
    """
    Pad a numpy array to the specified shape.

    Use val (default 0) to fill in the extra spaces.

    Parameters
    ----------
    arr : array of ints or floats
        The array to pad.
    shape : int, list of ints or tuple of ints
        The shape to which to pad ``a``.
        If any element of shape is 0, that size remains unchanged in
        that axis. If any element of shape is < 0, the size of ``a`` in that
        axis is incremented by the magnitude of that value.
    val : float
        The value with which to pad ``arr``. Default is 0.
    dtype : dtype
        The data type with which to pad ``arr``.

    Returns
    -------
    padded : array of ints or floats
        The padded version of ``arr``.
    """
    # For  padding a 1D array
    if isinstance(shape, int):
        if shape <= 0:
            rows = arr.size - shape
        else:
            rows = shape
            if rows < arr.size:
                print(' '.join(['arr.size is', str(arr.size),
                                ' but trying to pad to ',
                                str(rows), 'rows.']))
                return arr
        # Handle the case where a is a one-dimensional array
        padded = np.ones(rows, dtype=dtype) * val
        padded[:arr.size] = arr
        return padded

    # For padding arr n-D array
    new_shape = shape
    n_dim = len(shape)
    if n_dim > 4:
        print(''.join([str(n_dim), ' dimensions? Now you\'re getting greedy']))
        return arr

    for dim, _ in enumerate(shape):
        if shape[dim] <= 0:
            new_shape[dim] = arr.shape[dim] - shape[dim]
        else:
            if new_shape[dim] < arr.shape[dim]:
                print(''.join(['The variable shape in dimension ',
                               str(dim), ' is ', str(arr.shape[dim]),
                               ' but you are trying to pad to ',
                               str(new_shape[dim]), '.']))
                print('You aren\'t allowed to make it smaller.')
                return arr

    padded = np.ones(new_shape, dtype=dtype) * val
    if len(new_shape) == 2:
        padded[:arr.shape[0], :arr.shape[1]] = arr
        return padded
    if len(new_shape) == 3:
        padded[:arr.shape[0], :arr.shape[1], :arr.shape[2]] = arr
        return padded
    # A maximum of 4 dimensions is enforced.
    padded[:arr.shape[0], :arr.shape[1], :arr.shape[2], :arr.shape[3]] = arr
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
    sum_ = 0
    for i, character in enumerate(exp):
        sum_ += i + ord(character) + i * ord(character)
    return sum_


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
    min_ = int(np.mod(total_min, 60.))
    if precise:
        time_str = ' '.join([str(min_), 'min', time_str])
    else:
        time_str = ' '.join([str(min_), 'min'])

    # If necessary, calculate the total number of hours.
    total_hr = int(total_min / 60)
    if total_hr == 0:
        return time_str
    hr_ = int(np.mod(total_hr, 24.))
    if precise:
        time_str = ' '.join([str(hr_), 'hr', time_str])
    else:
        time_str = ' '.join([str(hr_), 'hr'])

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
    yr_ = int(total_mon / 12)
    if yr_ == 0:
        return time_str
    if precise:
        time_str = ' '.join([str(yr_), 'yr', time_str])
    else:
        time_str = ' '.join([str(yr_), 'yr'])

    return time_str


def fatigue(raw_activities, energies, fatigue_rate=3e-4, recharge_rate=1e-4):
    """
    Limit the frequency and intensity of activities with a model of fatigue.

    @param raw_activities: array of floats
        The activities before fatigue has been applied.
    @param energies: array of floats
        The accumulated energy a channel has at its disposal.
    @param fatigue_rate
        The rate at which energy is depleted when a channel is active.
    @param recharge_rate:
        The rate at which energy is re-accumulated.

    @return activities: array of floats
        The activities after fatigue has been applied.
    """

    energies -= fatigue_rate * raw_activities * energies
    energies += recharge_rate * (1. - raw_activities) * (1. - energies)
    activities = raw_activities * energies
    return activities


def format_decimals(array):
    """
    Format and print an array as a list of fixed decimal numbers in a string.

    Parameters
    ----------
    array : array of floats
        The array to be formatted.
    """
    if len(array.shape) == 2:
        for j in range(array.shape[1]):
            formatted = (' '.join(['{0},{1}:{2:.3}'.format(i, j, array[i, j])
                                   for i in range(array.shape[0])]))
            print(formatted)
    else:
        array = array.copy().ravel()
        formatted = (' '.join(['{0}:{1:.3}'.format(i, array[i])
                               for i in range(array.size)]))
        print(formatted)


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
    for localpath, _, filenames in os.walk(dir_name):
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
    plt.bone()
    img = plt.imshow(image_data)
    img.set_interpolation('nearest')
    plt.title(label)
    plt.xlabel('Max = {0:.3}, Min = {1:.3}'.format(np.max(image_data),
                                                   np.min(image_data)))
    fig.show()
    fig.canvas.draw()
