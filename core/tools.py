""" 
Constants and functions for use across the BECCA core
"""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Shared constants
EPSILON = sys.float_info.epsilon
BIG = 10 ** 20
MAX_INT16 = np.iinfo(np.int16).max

DARK_GREY = (0.2, 0.2, 0.2)
LIGHT_GREY = (0.9, 0.9, 0.9)
RED = (0.9, 0.3, 0.3)
# BECCA pallette
COPPER_HIGHLIGHT = (253./255., 249./255., 240./255.)
LIGHT_COPPER = (242./255., 166./255., 108./255.)
COPPER = (175./255., 102./255, 53./255.)
DARK_COPPER = (132./255., 73./255., 36./255.)
COPPER_SHADOW = (25./255., 22./255, 20./255.)
OXIDE = (20./255., 120./255., 150./255.)

def weighted_average(values, weights):
    """ 
    This function handles broadcasting a little more loosely than numpy.average
    """
    weighted_sum_values = np.sum(values * weights, axis=0) 
    sum_of_weights = np.sum(weights, axis=0) 
    return (weighted_sum_values / (sum_of_weights + EPSILON))[:,np.newaxis]

def generalized_mean(values, weights, exponent):
    """ 
    Raise values to a power before taking the weighted average.
    This can weight higher or lower values more heavily.
    At the extreme exponent of positive infinity, the 
    generalized mean becomes the maximum operator.
    At the extreme exponent of negative infinity, the 
    generalized mean becomes the minimum operator.
    """
    shifted_values = values + 1.
    values_to_power = shifted_values ** exponent
    mean_values_to_power = weighted_average(values_to_power, weights)
    shifted_mean = (mean_values_to_power + EPSILON) ** (1./exponent)
    mean = shifted_mean - 1.
    # Find means for which all weights are zero. These are undefined.
    # Set them equal to zero.
    sum_weights = np.sum(weights, axis=0)
    zero_indices = np.where(np.abs(sum_weights) < EPSILON)
    mean[zero_indices] = 0.
    return mean

def map_one_to_inf(a):
    """ 
    Map values from [0, 1] onto [0, inf) and 
    map values from [-1, 0] onto (-inf, 0] 
    """
    eps = np.finfo(np.double).eps
    a_prime = np.sign(a) / (1 - np.abs(a) + eps) - np.sign(a)
    return a_prime

def map_inf_to_one(a_prime):
    """ 
    Map values from [0, inf) onto [0, 1] and 
    map values from  (-inf, 0] onto [-1, 0] 
    """
    a = np.sign(a_prime) * (1 - 1 / (np.abs(a_prime) + 1))
    return a

def bounded_sum(a, axis=0):
    """ 
    Sum elements nonlinearly, such that the total is less than 1 
    
    To be more precise, as long as all elements in a are between -1
    and 1, their sum will also be between -1 and 1. a can be a 
    list or a numpy array. 
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

def pad(a, shape, val=0.):
    """
    Pad a numpy array to the specified shape
    
    If any element of shape is 0, that size remains unchanged in 
    that axis. If any element of shape is < 0, the size in that
    axis is incremented by the magnitude of that value.
    Use val (default 0) to fill in the extra spaces. 
    """
    if shape[0] <= 0:
        rows = a.shape[0] - shape[0]
    else:
        rows = shape[0]
        # debug
        if rows < a.shape[0]:
            print ' '.join(['a.shape[0] is', str(a.shape[0]), ' but trying to',
                            ' pad to ', str(rows), 'rows.'])
    if shape[1] <= 0:
        cols = a.shape[1] - shape[1]
    else:
        cols = shape[1]
        # debug
        if cols < a.shape[1]:
            print ' '.join(['a.shape[1] is', str(a.shape[1]), ' but trying to',
                            ' pad to ', str(cols), 'cols.'])
    padded = np.ones((rows,cols)) * val
    padded[:a.shape[0], :a.shape[1]] = a
    return padded

def str_to_int(exp):
    """ Convert a string to an integer """ 
    sum = 0
    for character in exp:
        sum += ord(character)
    return sum

def timestr(timestep, s_per_step=.25, precise=True): 
    """ Convert the number of time steps into an age """
    total_sec = timestep * s_per_step
    sec = int(np.mod(total_sec, 60.))
    time_str = ' '.join([str(sec), 'sec'])
    total_min = int(total_sec / 60)
    min = int(np.mod(total_min, 60.))
    if min == 0:
        return time_str
    if precise:
        time_str = ' '.join([str(min), 'min', time_str])
    else:
        time_str = ' '.join([str(min), 'min'])
    total_hr = int(total_min / 60)
    hr = int(np.mod(total_hr, 24.))
    if hr == 0:
        return time_str
    if precise:
        time_str = ' '.join([str(hr), 'hr', time_str])
    else:
        time_str = ' '.join([str(hr), 'hr'])
    total_day = int(total_hr / 24)
    day = int(np.mod(total_day, 30.))
    if day == 0:
        return time_str
    if precise:
        time_str = ' '.join([str(day), 'days'])
    else:
        time_str = ' '.join([str(day), 'days', time_str])
    total_mon = int(total_day / 30)
    mon = int(np.mod(total_mon, 12.))
    if mon == 0:
        return time_str
    if precise:
        time_str = ' '.join([str(mon), 'mon'])
    else:
        time_str = ' '.join([str(mon), 'mon', time_str])
    yr = int(total_mon / 12)
    if yr == 0:
        return time_str
    if precise: 
        time_str = ' '.join([str(yr), 'yr'])
    else:
        time_str = ' '.join([str(yr), 'yr', time_str])
    return time_str

def get_files_with_suffix(dir_name, suffixes):
    """ Get all of the files with a given suffix in dir recursively """
    found_filenames = []
    for localpath, directories, filenames in os.walk(dir_name):
        for filename in filenames:
            for suffix in suffixes:
                if filename.endswith(suffix):
                    found_filenames.append(os.path.join(localpath, filename))
    found_filenames.sort()
    return found_filenames

def visualize_array(image_data, shape=None, save_eps=False, 
                    label='data_figure', epsfilename=None, show=True):
    """ Produce a visual representation of the image_data matrix """    
    if shape is None:
        shape = image_data.shape
    if epsfilename is None:
        epsfilename = 'log/' + label + '.eps'
    fig = plt.figure(str_to_int(label))
    
    # Diane made the brilliant suggestion to leave this plot in color. 
    # It looks much prettier.
    plt.summer()
    im = plt.imshow(image_data[0:shape[0], 0:shape[1]])
    im.set_interpolation('nearest')
    plt.title(label)
    if show:
        fig.show()
    fig.canvas.draw()
    if save_eps:
        fig.savefig(epsfilename, format='eps')
    return

def visualize_hub(hub, show=False):
    """ Give a visual update of the internal workings of the hub """
    # Plot reward value
    fig = plt.figure(311)
    fig.clf()
    plt.subplot(1,2,1)
    plt.gray()
    plt.imshow(hub.reward.astype(np.float), interpolation='nearest', 
               vmin=-1., vmax=1.)
    plt.title('reward')
    plt.subplot(1,2,2)
    plt.gray()
    plt.imshow((np.log(hub.count + 1.)).astype(np.float), 
                interpolation='nearest')
    plt.title(''.join(['count, max = ', str(int(np.max(hub.count)))]))
    frames_directory = os.path.join('core', 'hub_frames')
    filename =  ''.join([hub.name, '_reward_', 
                         str(hub.frame_counter), '.png'])
    full_filename = os.path.join(frames_directory, filename)
    dpi = 80 # for a resolution of 720 lines
    #dpi = 120 # for a resolution of 1080 lines
    #dpi = 480 # for a resolution of 4320 lines
    plt.savefig(full_filename, format='png', dpi=dpi, 
                facecolor=fig.get_facecolor(), edgecolor='none') 
    hub.frame_counter += 1
    if show:
        plt.show()
            
def visualize_hierarchy(agent, show=True):
    """ Show how cables compose bundles across gearboxes """
    fig = plt.figure(num=84)
    fig.clf()
    width = 9.
    depth = width * 9. / 16.
    fig, ax = plt.subplots(num=84, figsize=(width, depth))

    b = 1. # border
    hgt_lo = -b
    hgt_hi = depth + b
    wid_lo = -b
    wid_hi = width + b
    ax.fill([wid_lo, wid_hi, wid_hi, wid_lo, wid_lo], 
            [hgt_lo, hgt_lo, hgt_hi, hgt_hi, hgt_lo], 
            color=COPPER_HIGHLIGHT, zorder=-10)

    dt = agent.drivetrain
    n_gb = len(dt.gearboxes)
    max_gearboxes = 4
    y = 0.
    delta_y = depth  / float(max_gearboxes - 1)
    i_gearbox_arborkey, i_cable_arborkey = dt.map_index(agent.arborkey_goal)
    i_gearbox_hub, i_cable_hub = dt.map_index(agent.hub_goal)
    i_gearbox_attended, i_cable_attended = dt.map_index(agent.attended_index)
    recently_attended = agent.mainspring.stm_indices[-1]
    decayed_activities = agent.mainspring.get_decayed_activities(-1)

    unmasked = [[]]
    for i_mask in np.arange(agent.hub.mask.size):
        i_gearbox_mask, i_cable_mask = dt.map_index(i_mask)
        if len(unmasked) < i_gearbox_mask + 1:
            unmasked.append([])
        if agent.hub.mask[i_mask] > 0.:
            unmasked[i_gearbox_mask].append(i_cable_mask)
    for i_gb in np.arange(n_gb):
        # Plot the cable activities and goals
        gb = dt.gearboxes[i_gb]
        cables = gb.cable_activities
        goals = gb.cable_goals
        bundles = gb.bundle_activities
        n_cables = cables.size
        n_bundles = bundles.size
        for i_cable in np.arange(n_cables):
            if i_cable in unmasked[i_gb]:
                x = width * i_cable / float(n_cables) 
                cable_diameter = .1 + 5. * cables[int(i_cable)] ** .5
                goal_diameter = .1 + 5. * goals[int(i_cable)] ** .5
                plt.plot(x, y, '.', color=COPPER_SHADOW, 
                         markersize=cable_diameter)
                plt.plot(x, y, '.', color=OXIDE, 
                         markersize=goal_diameter, zorder=-8)
        
        # Mark the arborkey goal feature
        if i_gearbox_arborkey is not None:
            if i_gb == i_gearbox_arborkey:
                x = width * i_cable_arborkey / float(n_cables) 
                arbor_diameter = 7.
                plt.plot(x, y, 's', color=COPPER_SHADOW, 
                         markerfacecolor='none', markersize=arbor_diameter)
        # Mark the hub goal feature
        if i_gearbox_hub is not None:
            if i_gb == i_gearbox_hub:
                x = width * i_cable_hub / float(n_cables) 
                hub_diameter = 4.
                plt.plot(x, y, 's', color=COPPER_SHADOW, 
                         markerfacecolor='none', markersize=hub_diameter)
        # Mark the attended feature
        if i_gearbox_attended is not None:
            if i_gb == i_gearbox_attended:
                x = width * i_cable_attended / float(n_cables) 
                attended_diameter = 6.
                plt.plot(x, y, 'o', color=COPPER_SHADOW, 
                         markerfacecolor='none', markersize=attended_diameter)
                plt.plot(x, y, '+', color=COPPER_SHADOW, 
                         markersize=attended_diameter)

        # Mark recently attended features
        for count_stm, i_stm in enumerate(recently_attended):
            i_gearbox_stm, i_cable_stm = dt.map_index(i_stm)
            if i_gb == i_gearbox_stm:
                x = width * i_cable_stm / float(n_cables) 
                alpha = decayed_activities[count_stm]
                attended_diameter = 4.
                plt.plot(x, y, 'o', color=COPPER_SHADOW, alpha=alpha, 
                         markerfacecolor='none', markersize=attended_diameter)

        # Show bundle membership
        for i_bundle in np.arange(n_bundles):
            x_bundle = width * float(i_bundle) / float(n_bundles)
            projection = gb.get_index_projection(i_bundle)
            pre_members = np.where(projection[:,0] > 0.)[0]
            for member in pre_members:
                x_cable = width * float(member) / float(n_cables)
                plt.plot(np.array([x_bundle, x_cable - .01]),
                         np.array([y + delta_y, y]), linewidth = .1,
                         color=COPPER_SHADOW)
            post_members = np.where(projection[:,1] > 0.)[0]
            for member in post_members:
                x_cable = width * float(member) / float(n_cables)
                plt.plot(np.array([x_bundle + .03, x_cable + .01]),
                         np.array([y + delta_y, y]), linewidth = .1,
                         color=COPPER)
        y += delta_y

    n_cables_0 = dt.gearboxes[0].cable_activities.size 
    x_divider = (float(agent.num_actions) - .5) * width / float(n_cables_0) 
    plt.plot(np.array([x_divider, x_divider]), 
             np.array([-.6,0.]), linewidth=.5,
             color=COPPER_SHADOW)
    plt.text(x_divider - .05, -.1, 'actions', 
             horizontalalignment='right',
             verticalalignment='top',
             rotation='vertical',
             fontsize=5., 
             color=COPPER_SHADOW)
    plt.text(x_divider + .05, -.1, 'sensors', 
             horizontalalignment='left',
             verticalalignment='top',
             rotation='vertical',
             fontsize=5., 
             color=COPPER_SHADOW)
    #plt.text(width, -.2, ' '.join([str(agent.timestep), 'time steps']), 
    plt.text(width, -.2, timestr(agent.timestep), 
             horizontalalignment='right',
             verticalalignment='top',
             rotation='horizontal',
             fontsize=9., 
             color=COPPER_SHADOW)

    plt.axis('equal')
    plt.axis('off')
    fig.canvas.draw()
    # Save the control panel image
    filename =  ''.join([agent.name, '_hierarchy_', 
                         str(agent.frame_counter), '.png'])
    full_filename = os.path.join(agent.frames_directory, filename)
    agent.frame_counter += 1
    #dpi = 80 # for a resolution of 720 lines
    #dpi = 120 # for a resolution of 1080 lines
    dpi = 480 # for a resolution of 4320 lines
    plt.savefig(full_filename, format='png', dpi=dpi, 
                facecolor=fig.get_facecolor(), edgecolor='none') 
    if show:
        plt.show()
