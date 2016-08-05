"""
Connect a world with a brain and set it to running.
"""
import numpy as np

from becca.brain import Brain

def run(world, restore=False):
    """
    Run Becca with a world.

    Connect the brain and the world together and runs them for as long
    as the world dictates.

    Parameters
    ----------
    restore : bool, optional
        If ``restore`` is True, try to restore the brain
        from a previously saved
        version, picking up where it left off.
        Otherwise it create a new one. The default is False.

    Returns
    -------
    performance : float
        The performance of the brain over its lifespan, measured by the
        average reward it gathered per time step.
    """
    brain_name = '{0}_brain'.format(world.name)
    brain = Brain(world.num_sensors,
                  world.num_actions,
                  brain_name=brain_name)
    if restore:
        brain = brain.restore()
    # Start at a resting state.
    actions = np.zeros(world.num_actions)
    sensors, reward = world.step(actions)
    # Repeat the loop through the duration of the existence of the world:
    # sense, act, repeat.
    while world.is_alive():
        actions = brain.sense_act_learn(sensors, reward)
        sensors, reward = world.step(actions)
        world.visualize(brain)
    # Wrap up the run.
    performance = brain.report_performance()
    return performance
