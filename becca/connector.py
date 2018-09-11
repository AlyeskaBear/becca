"""
Connect a world with a brain and set it to running.
"""
import copy

import numpy as np

from becca.brain import Brain


def run(world, config):
    """
    Run Becca with a world.

    Connect the brain and the world together and run them for as long
    as the world dictates.

    Parameters
    ----------
    world : World
        The world that Becca will learn.
        See the world.py documentation for a full description.
    config: dict
        Keys are configurable brain parameters and values are their
        desired values. See configure() in becca/brain.py for
        a full list.

    Returns
    -------
    performance : float
        The performance of the brain over its lifespan, measured by the
        average reward it gathered per time step.
    """
    brain = Brain(world, config)

    # Start at a resting state.
    actions = np.zeros(world.n_actions)
    sensors, reward = world.step(actions)

    # Repeat the loop through the duration of the existence of the world:
    # sense, act, repeat.
    while world.is_alive():
        actions = brain.sense_act_learn(copy.deepcopy(sensors), reward)
        sensors, reward = world.step(copy.copy(actions))

    # Wrap up the run.
    try:
        world.close_world(brain)
    except AttributeError:
        print("Closing", world.name)

    performance = brain.report_performance()
    return performance
