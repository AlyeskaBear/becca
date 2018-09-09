"""
Connect a world with a brain and set it to running.
"""
import copy

import numpy as np

from becca.brain import Brain
import becca_viz.viz as viz


def run(
    world,
    full_visualization=False,
    restore=False,
    visualize_interval=1e3,
):
    """
    Run Becca with a world.

    Connect the brain and the world together and run them for as long
    as the world dictates.

    Parameters
    ----------
    world : World
        The world that Becca will learn.
        See the world.py documentation for a full description.
    full_visualize: bool
        Flag indicating whether to do a full visualization
        of the becca brain.
    restore : bool, optional
        If restore is True, try to restore the brain
        from a previously saved
        version, picking up where it left off.
        Otherwise it create a new one. The default is False.
    visualize_interval: int
        The number of time steps between creating a new performance
        calculation and visualization of the brain.

    Returns
    -------
    performance : float
        The performance of the brain over its lifespan, measured by the
        average reward it gathered per time step.
    """
    brain_name = '{0}_brain'.format(world.name)

    try:
        brain = Brain(
            n_sensors=world.num_sensors,
            n_actions=world.num_actions,
            brain_name=brain_name,
            log_directory=world.log_directory,
        )
    # Catch the case where world has no log_directory.
    except AttributeError:
        brain = Brain(
            n_sensors=world.num_sensors,
            n_actions=world.num_actions,
            brain_name=brain_name,
        )

    if restore:
        brain = brain.restore()

    # Start at a resting state.
    actions = np.zeros(world.num_actions)
    sensors, reward = world.step(actions)

    # Repeat the loop through the duration of the existence of the world:
    # sense, act, repeat.
    while world.is_alive():
        actions = brain.sense_act_learn(copy.deepcopy(sensors), reward)
        sensors, reward = world.step(copy.copy(actions))

        # Create visualizations.
        if brain.timestep % visualize_interval == 0:
            viz.visualize(brain, full_visualization=full_visualization)
        # if world.timestep % world.visualize_interval == 0:
        #     world.visualize()

    # Wrap up the run.
    try:
        world.close_world(brain)
    except AttributeError:
        print("Closing", world.name_long)

    performance = brain.report_performance()
    return performance
